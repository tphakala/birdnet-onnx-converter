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


def test_fp16_conversion_no_rewires_on_dual_role():
    """End-to-end: after decoupling, FP16 conversion needs zero mixed-type rewires
    and the converted model has no mixed-precision MatMul."""
    from optimize import convert_to_fp16, fix_fp16_type_mismatches

    fp16 = convert_to_fp16(_make_dual_role_model())
    assert fp16 is not None
    onnx.checker.check_model(fp16)
    assert [o.name for o in fp16.graph.output] == ["pred", "emb"]

    # A second sentinel pass must find nothing left to rewire.
    rewires, _vi_fixes = fix_fp16_type_mismatches(fp16)
    assert rewires == 0, "decoupling should leave no mixed-type ops to rewire"

    # No MatMul should mix fp16 and fp32 inputs.
    init_t = {i.name: i.data_type for i in fp16.graph.initializer}
    vi_t = {v.name: v.type.tensor_type.elem_type for v in fp16.graph.value_info}
    fp16_t, fp32_t = onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT
    for node in fp16.graph.node:
        if node.op_type != "MatMul":
            continue
        types = {init_t.get(i, vi_t.get(i)) for i in node.input}
        assert not (fp16_t in types and fp32_t in types), f"mixed-type MatMul: {node.name}"
