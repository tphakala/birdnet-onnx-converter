"""Tests for ONNX model optimization."""

from pathlib import Path

import onnx


def test_optimization_reduces_ops(converted_onnx_path: Path):
    """Verify optimization reduces operation count."""
    from optimize import count_ops, optimize_model

    # Load original model
    original_model = onnx.load(str(converted_onnx_path))
    initial_ops = count_ops(original_model)

    # Run optimization
    optimized = optimize_model(str(converted_onnx_path))
    final_ops = count_ops(optimized)

    # Verify reduction
    assert sum(final_ops.values()) < sum(initial_ops.values()), (
        f"Optimization should reduce ops: {sum(initial_ops.values())} -> {sum(final_ops.values())}"
    )


def test_optimization_removes_dft(converted_onnx_path: Path):
    """Verify DFT operations are replaced with MatMul."""
    from optimize import count_ops, optimize_model

    original_model = onnx.load(str(converted_onnx_path))
    initial_dft = count_ops(original_model).get("DFT", 0)

    optimized = optimize_model(str(converted_onnx_path))
    final_dft = count_ops(optimized).get("DFT", 0)

    assert final_dft == 0, f"All DFT ops should be removed: {initial_dft} -> {final_dft}"


def test_optimization_removes_casts(converted_onnx_path: Path):
    """Verify Cast operations are removed (requires onnx_ir)."""
    import pytest

    from optimize import count_ops, optimize_model

    # Check if onnx_ir is available (needed for INT32->INT64 conversion)
    try:
        import onnx_ir  # noqa: F401

        onnx_ir_available = True
    except (ImportError, AttributeError):
        onnx_ir_available = False

    optimized = optimize_model(str(converted_onnx_path))
    final_casts = count_ops(optimized).get("Cast", 0)

    if not onnx_ir_available:
        pytest.skip("onnx_ir not available - Cast removal requires INT32->INT64 conversion")

    assert final_casts == 0, f"All Cast ops should be removed, found {final_casts}"


def test_optimized_model_is_valid(optimized_fp32_path: Path):
    """Verify optimized model passes ONNX checker."""
    model = onnx.load(str(optimized_fp32_path))
    onnx.checker.check_model(model)


def test_fp16_conversion(optimized_fp32_path: Path):
    """Verify FP16 conversion produces valid model."""
    from optimize import convert_to_fp16

    model = onnx.load(str(optimized_fp32_path))
    fp16_model = convert_to_fp16(model)

    assert fp16_model is not None, "FP16 conversion should succeed"
    onnx.checker.check_model(fp16_model)


def test_fp16_model_is_valid(optimized_fp16_path: Path):
    """Verify FP16 model passes ONNX checker."""
    model = onnx.load(str(optimized_fp16_path))
    onnx.checker.check_model(model)


def _make_dual_role_model() -> onnx.ModelProto:
    """Build a tiny model whose 'emb' tensor is both a graph output and the input
    to an internal MatMul (the dual-role case that keep_io_types corrupts)."""
    import numpy as np
    import onnx.helper as h
    from onnx import TensorProto

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    emb = h.make_tensor_value_info("emb", TensorProto.FLOAT, [1, 4])
    pred = h.make_tensor_value_info("pred", TensorProto.FLOAT, [1, 4])
    eye = np.eye(4, dtype=np.float32).ravel()
    w1 = h.make_tensor("w1", TensorProto.FLOAT, [4, 4], eye)
    w2 = h.make_tensor("w2", TensorProto.FLOAT, [4, 4], eye)
    nodes = [
        h.make_node("MatMul", ["X", "w1"], ["emb"]),  # emb is a graph output...
        h.make_node("MatMul", ["emb", "w2"], ["pred"]),  # ...and consumed here
    ]
    graph = h.make_graph(nodes, "dual_role", [x], [pred, emb], [w1, w2])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9
    return model


def test_decouple_dual_role_outputs_count_and_names():
    """Decoupling rewrites the dual-role output via Identity while preserving the
    external output name; terminal-only outputs are left untouched."""
    from optimize import decouple_dual_role_outputs

    model = _make_dual_role_model()
    decoupled = decouple_dual_role_outputs(model)

    assert decoupled == 1, "only the dual-role 'emb' output should be decoupled"
    assert [o.name for o in model.graph.output] == ["pred", "emb"], (
        "external output names must be preserved for name-based consumers"
    )
    # emb is now produced by an Identity, and the original producer feeds a
    # private internal tensor that the classifier MatMul consumes.
    identity = [n for n in model.graph.node if n.op_type == "Identity"]
    assert len(identity) == 1 and identity[0].output[0] == "emb"


def test_decouple_is_noop_on_terminal_outputs():
    """A model whose outputs are not consumed internally must be left unchanged."""
    import numpy as np
    import onnx.helper as h
    from onnx import TensorProto

    from optimize import decouple_dual_role_outputs

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = h.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    w = h.make_tensor("w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).ravel())
    graph = h.make_graph([h.make_node("MatMul", ["X", "w"], ["Y"])], "g", [x], [y], [w])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9

    assert decouple_dual_role_outputs(model) == 0


def _count_mixed_type_matmuls(model: onnx.ModelProto) -> int:
    """Count MatMul nodes that mix fp16 and fp32 inputs.

    Runs shape inference first so intermediate tensors (e.g. Cast outputs) carry
    a resolved element type; without it those tensors resolve to None and the
    scan is vacuous (it would never see the fp32 side of the mismatch).
    """
    model = onnx.shape_inference.infer_shapes(model)
    types: dict[str, int] = {}
    for init in model.graph.initializer:
        types[init.name] = init.data_type
    for collection in (model.graph.value_info, model.graph.input, model.graph.output):
        for value in collection:
            types[value.name] = value.type.tensor_type.elem_type

    fp16_t, fp32_t = onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT
    mixed = 0
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        input_types = {types.get(i) for i in node.input}
        if fp16_t in input_types and fp32_t in input_types:
            mixed += 1
    return mixed


def test_fp16_decouple_prevents_mixed_type_matmul():
    """Counterfactual regression guard: raw keep_io_types FP16 conversion of a
    dual-role model produces a mixed-type MatMul; running the decouple first
    eliminates it. This test FAILS if decouple_dual_role_outputs is reverted."""
    from onnxconverter_common import float16

    from optimize import decouple_dual_role_outputs

    # Without decoupling, the bug is present (and detectable).
    without = float16.convert_float_to_float16(_make_dual_role_model(), keep_io_types=True)
    assert _count_mixed_type_matmuls(without) > 0, (
        "expected the dual-role bug to be present without decoupling"
    )

    # With decoupling, no MatMul mixes precisions.
    decoupled = _make_dual_role_model()
    decouple_dual_role_outputs(decoupled)
    fixed = float16.convert_float_to_float16(decoupled, keep_io_types=True)
    assert _count_mixed_type_matmuls(fixed) == 0, "decoupling should remove all mixed-type MatMuls"


def test_fp16_conversion_preserves_outputs_and_values_on_dual_role():
    """End-to-end convert_to_fp16: output names preserved, sentinel finds zero
    rewires, and FP16 outputs match FP32 within tolerance."""
    import numpy as np
    import onnxruntime as ort

    from optimize import convert_to_fp16, fix_fp16_type_mismatches

    fp32 = _make_dual_role_model()
    x = np.arange(4, dtype=np.float32).reshape(1, 4)
    ref = ort.InferenceSession(fp32.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        None, {"X": x}
    )

    fp16 = convert_to_fp16(_make_dual_role_model())
    assert fp16 is not None
    onnx.checker.check_model(fp16)
    assert [o.name for o in fp16.graph.output] == ["pred", "emb"], "output names preserved"

    rewires, _vi_fixes = fix_fp16_type_mismatches(fp16)
    assert rewires == 0, "decoupling should leave no mixed-type ops to rewire"
    assert _count_mixed_type_matmuls(fp16) == 0

    out = ort.InferenceSession(fp16.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        None, {"X": x}
    )
    for ref_o, fp16_o in zip(ref, out):
        np.testing.assert_allclose(ref_o, fp16_o, atol=1e-2)


def test_decouple_avoids_name_collision():
    """A pre-existing tensor named like the minted internal name must not be
    clobbered; decouple bumps the suffix instead."""
    import onnx.helper as h

    from optimize import decouple_dual_role_outputs

    model = _make_dual_role_model()
    # Seed a name that collides with the default minted internal name "emb_fp16src".
    model.graph.initializer.append(h.make_tensor("emb_fp16src", onnx.TensorProto.FLOAT, [1], [0.0]))

    assert decouple_dual_role_outputs(model) == 1
    node_outputs = {o for n in model.graph.node for o in n.output}
    assert "emb_fp16src2" in node_outputs, "collision should bump to emb_fp16src2"
    init_names = {i.name for i in model.graph.initializer}
    assert "emb_fp16src" in init_names, "pre-existing tensor must be left intact"


def test_decouple_skips_output_with_no_producer():
    """A graph output that is also a graph input (no producing node) has no fp16
    boundary to corrupt and must be skipped."""
    import numpy as np
    import onnx.helper as h
    from onnx import TensorProto

    from optimize import decouple_dual_role_outputs

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    pred = h.make_tensor_value_info("pred", TensorProto.FLOAT, [1, 4])
    w = h.make_tensor("w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).ravel())
    # X is both a graph input and a graph output (passthrough), and is consumed.
    graph = h.make_graph([h.make_node("MatMul", ["X", "w"], ["pred"])], "g", [x], [pred, x], [w])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9

    assert decouple_dual_role_outputs(model) == 0
