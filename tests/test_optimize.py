"""Tests for ONNX model optimization."""

from pathlib import Path

import onnx
import pytest


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


def _make_preproc_conv_model() -> onnx.ModelProto:
    """Build a tiny model with a preprocessing MatMul feeding a backbone Conv,
    mirroring the spectrogram-front-end -> CNN structure of BirdNET."""
    import numpy as np
    import onnx.helper as h
    from onnx import TensorProto

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    out = h.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    w_spec = h.make_tensor("w_spec", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).ravel())
    w_conv = h.make_tensor("w_conv", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
    s_img = h.make_tensor("s_img", TensorProto.INT64, [4], [1, 1, 2, 2])
    s_out = h.make_tensor("s_out", TensorProto.INT64, [2], [1, 4])
    nodes = [
        h.make_node("MatMul", ["X", "w_spec"], ["spec"], name="pre_matmul"),
        h.make_node("Reshape", ["spec", "s_img"], ["img"], name="pre_reshape"),
        h.make_node("Conv", ["img", "w_conv"], ["feat"], name="backbone_conv"),
        h.make_node("Reshape", ["feat", "s_out"], ["out"], name="post_reshape"),
    ]
    graph = h.make_graph(nodes, "preproc_conv", [x], [out], [w_spec, w_conv, s_img, s_out])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9
    return model


def test_find_preprocessing_node_names_identifies_frontend():
    """The front-end is the ancestors of the first Conv; the Conv and downstream
    nodes are excluded."""
    from optimize import find_preprocessing_node_names

    names = find_preprocessing_node_names(_make_preproc_conv_model())
    assert names == {"pre_matmul", "pre_reshape"}
    assert "backbone_conv" not in names
    assert "post_reshape" not in names


def test_find_preprocessing_node_names_empty_without_conv():
    """A graph with no Conv has no locatable front-end."""
    from optimize import find_preprocessing_node_names

    assert find_preprocessing_node_names(_make_dual_role_model()) == set()


def test_find_preprocessing_node_names_uses_first_2d_conv():
    """With multiple 2-D Convs, only ancestors of the FIRST one are the front-end;
    nodes between the first and a later Conv are downstream and excluded."""
    import onnx.helper as h
    from onnx import TensorProto

    from optimize import find_preprocessing_node_names

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    out = h.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 4, 4])
    w = h.make_tensor("w", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
    s = h.make_tensor("s", TensorProto.FLOAT, [1], [2.0])
    nodes = [
        h.make_node("Mul", ["X", "s"], ["pre"], name="pre_mul"),  # front-end
        h.make_node("Conv", ["pre", "w"], ["c1"], name="conv1"),  # backbone stem
        h.make_node("Relu", ["c1"], ["r"], name="between_relu"),  # downstream of conv1
        h.make_node("Conv", ["r", "w"], ["out"], name="conv2"),
    ]
    graph = h.make_graph(nodes, "multi_conv", [x], [out], [w, s])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9

    names = find_preprocessing_node_names(model)
    assert names == {"pre_mul"}
    assert {"conv1", "between_relu", "conv2"}.isdisjoint(names)


def test_find_preprocessing_node_names_protects_1d_frontend_conv():
    """A 1-D Conv on the waveform (the folded spectrogram) is front-end: the anchor
    is the first Conv with a rank-4 weight, so the 1-D Conv and its neighbours are
    ancestors of the stem and protected, while the stem itself is not."""
    import numpy as np
    import onnx.helper as h
    from onnx import TensorProto, numpy_helper

    from optimize import find_preprocessing_node_names

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])
    out = h.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 2, 2])
    inits = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes"),
        numpy_helper.from_array(np.ones((2, 1, 4), dtype=np.float32), name="w_fold"),
        numpy_helper.from_array(np.array([1, 1, 2, 7], dtype=np.int64), name="shape_img"),
        numpy_helper.from_array(np.ones((2, 1, 1, 1), dtype=np.float32), name="w_stem"),
    ]
    nodes = [
        h.make_node("Unsqueeze", ["X", "axes"], ["x3"], name="fold_unsqueeze"),
        h.make_node("Conv", ["x3", "w_fold"], ["c1"], name="fold_conv", strides=[2]),
        h.make_node("Transpose", ["c1"], ["c1t"], name="fold_transpose", perm=[0, 2, 1]),
        h.make_node("Reshape", ["c1t", "shape_img"], ["img"], name="fold_reshape"),
        h.make_node("Conv", ["img", "w_stem"], ["out"], name="stem_conv"),
    ]
    graph = h.make_graph(nodes, "fold_frontend", [x], [out], inits)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9

    names = find_preprocessing_node_names(model)
    assert {"fold_unsqueeze", "fold_transpose", "fold_reshape"} <= names
    assert "stem_conv" not in names


def test_find_preprocessing_node_names_names_unnamed_frontend_nodes():
    """An unnamed front-end node is given a name in place and included, so the
    name-based block list cannot silently leave it in FP16."""
    import onnx.helper as h
    from onnx import TensorProto

    from optimize import find_preprocessing_node_names

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    out = h.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 4, 4])
    w = h.make_tensor("w", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
    s = h.make_tensor("s", TensorProto.FLOAT, [1], [2.0])
    nodes = [
        h.make_node("Mul", ["X", "s"], ["pre"]),  # no name
        h.make_node("Conv", ["pre", "w"], ["out"], name="conv1"),
    ]
    graph = h.make_graph(nodes, "unnamed_pre", [x], [out], [w, s])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9

    names = find_preprocessing_node_names(model)
    assert len(names) == 1
    minted = next(iter(names))
    assert minted.startswith("_fp32_preproc_")
    # the name was assigned in place on the actual node
    mul = next(n for n in model.graph.node if n.op_type == "Mul")
    assert mul.name == minted


def test_fp16_keeps_preprocessing_in_fp32_by_default():
    """The preprocessing MatMul weight stays FP32 while the backbone Conv weight
    is converted to FP16."""
    from optimize import convert_to_fp16

    fp16 = convert_to_fp16(_make_preproc_conv_model())
    assert fp16 is not None
    onnx.checker.check_model(fp16)
    dtypes = {i.name: i.data_type for i in fp16.graph.initializer}
    assert dtypes["w_spec"] == onnx.TensorProto.FLOAT, "preprocessing MatMul must stay FP32"
    assert dtypes["w_conv"] == onnx.TensorProto.FLOAT16, "backbone Conv must be FP16"


def test_fp16_full_converts_preprocessing_to_fp16():
    """keep_preprocessing_fp32=False (CLI --fp16-full) converts the whole graph."""
    from optimize import convert_to_fp16

    fp16 = convert_to_fp16(_make_preproc_conv_model(), keep_preprocessing_fp32=False)
    assert fp16 is not None
    onnx.checker.check_model(fp16)
    dtypes = {i.name: i.data_type for i in fp16.graph.initializer}
    assert dtypes["w_spec"] == onnx.TensorProto.FLOAT16
    assert dtypes["w_conv"] == onnx.TensorProto.FLOAT16


def _make_conv_swish_model() -> onnx.ModelProto:
    """Tiny Conv -> Swish (Sigmoid * Mul) -> scale Mul graph. The trailing Mul is
    NOT sigmoid-gated, so it must not be treated as an activation."""
    import onnx.helper as h
    from onnx import TensorProto

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    out = h.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 4, 4])
    w = h.make_tensor("w_conv", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
    scale = h.make_tensor("scale_const", TensorProto.FLOAT, [1], [0.5])
    nodes = [
        h.make_node("Conv", ["X", "w_conv"], ["c"], name="conv1"),
        h.make_node("Sigmoid", ["c"], ["s"], name="act_sigmoid"),
        h.make_node("Mul", ["c", "s"], ["sw"], name="act_mul"),  # Swish (sigmoid-gated)
        h.make_node("Mul", ["sw", "scale_const"], ["out"], name="scale_mul"),  # NOT gated
    ]
    graph = h.make_graph(nodes, "conv_swish", [x], [out], [w, scale])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9
    return model


def test_find_swish_node_names_detects_only_gated_muls():
    """Only the sigmoid-gated Mul and its Sigmoid are activations; an unrelated
    (scale) Mul and the Conv are not."""
    from optimize import find_swish_node_names

    names = find_swish_node_names(_make_conv_swish_model())
    assert names == {"act_mul", "act_sigmoid"}
    assert {"scale_mul", "conv1"}.isdisjoint(names)


def test_fp16_keeps_activations_fp16_by_default():
    """Default conversion converts the Swish activation (Sigmoid, Mul) to FP16."""
    from optimize import convert_to_fp16

    fp16 = convert_to_fp16(_make_conv_swish_model())
    assert fp16 is not None
    onnx.checker.check_model(fp16)
    dtypes = {i.name: i.data_type for i in fp16.graph.initializer}
    assert dtypes["w_conv"] == onnx.TensorProto.FLOAT16, "backbone Conv must be FP16"


def test_fp16_keep_activations_fp32_blocks_only_swish():
    """keep_activations_fp32=True keeps the Swish Sigmoid/Mul in FP32 (so ORT's CPU
    EP can fuse QuickGelu) while conv weights AND unrelated Muls still go FP16."""
    from optimize import convert_to_fp16

    default = convert_to_fp16(_make_conv_swish_model())
    kept = convert_to_fp16(_make_conv_swish_model(), keep_activations_fp32=True)
    assert default is not None and kept is not None
    onnx.checker.check_model(kept)

    init = {i.name: i.data_type for i in kept.graph.initializer}
    # Conv weights stay FP16 (RAM benefit preserved).
    assert init["w_conv"] == onnx.TensorProto.FLOAT16, "backbone Conv must be FP16"
    # The non-activation (scale) Mul is NOT over-blocked: its constant goes FP16.
    assert init["scale_const"] == onnx.TensorProto.FLOAT16, (
        "an unrelated Mul must not be forced to FP32"
    )

    # Keeping the activation in FP32 inserts FP16<->FP32 boundary Casts that the
    # default all-FP16 conversion does not need.
    def cast_count(m: onnx.ModelProto) -> int:
        return sum(1 for n in m.graph.node if n.op_type == "Cast")

    assert cast_count(kept) > cast_count(default), "activation boundary casts expected"

    # Both halves of the Swish pair (Sigmoid and the gated Mul) run in FP32, so
    # ORT can fuse them; checking only the Sigmoid would miss a Mul-only regression.
    vi = {v.name: v.type.tensor_type.elem_type for v in kept.graph.value_info}
    sig = next(n for n in kept.graph.node if n.op_type == "Sigmoid")
    assert vi.get(sig.output[0]) == onnx.TensorProto.FLOAT, "Sigmoid must stay FP32"
    act_mul = next(n for n in kept.graph.node if n.name == "act_mul")
    assert vi.get(act_mul.output[0]) == onnx.TensorProto.FLOAT, "gated Mul must stay FP32"


def test_fp16_protects_spectrogram_matmuls_on_real_model(optimized_fp32_path: Path):
    """On the real optimized BirdNET model, the spectrogram MatMuls (RFFT and mel
    filterbank) keep FP32 weights with protection and become FP16 without it."""
    from optimize import convert_to_fp16, find_preprocessing_node_names

    base = onnx.load(str(optimized_fp32_path))
    preprocessing = find_preprocessing_node_names(base)
    spectrogram_matmuls = {
        n.name for n in base.graph.node if n.op_type == "MatMul" and n.name in preprocessing
    }
    assert spectrogram_matmuls, "expected to find spectrogram MatMuls in the front-end"

    def matmul_weight_dtypes(model: onnx.ModelProto, names: set[str]) -> set[int]:
        init_types = {i.name: i.data_type for i in model.graph.initializer}
        dtypes = set()
        for node in model.graph.node:
            if node.name in names:
                dtypes.update(init_types[i] for i in node.input if i in init_types)
        return dtypes

    protected = convert_to_fp16(onnx.load(str(optimized_fp32_path)), keep_preprocessing_fp32=True)
    full = convert_to_fp16(onnx.load(str(optimized_fp32_path)), keep_preprocessing_fp32=False)
    assert protected is not None and full is not None

    assert onnx.TensorProto.FLOAT in matmul_weight_dtypes(protected, spectrogram_matmuls), (
        "spectrogram MatMul weights must remain FP32 when protected"
    )
    assert matmul_weight_dtypes(full, spectrogram_matmuls) == {onnx.TensorProto.FLOAT16}, (
        "spectrogram MatMul weights must be FP16 with --fp16-full"
    )

    # The protected model must load and run in ONNX Runtime. The FP32/FP16
    # block-list boundary leaves stale value_info that ORT rejects unless the
    # value_info sentinel fixes it; this guards that the sentinel keeps running.
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(
        protected.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    spec = session.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
    session.run(None, {spec.name: np.zeros(shape, dtype=np.float32)})


def test_prepare_for_quantization_excludes_preprocessing_and_refreshes(tmp_path: Path):
    """prepare_for_quantization returns the front-end node names and writes a
    model whose stale value_info has been cleared and re-inferred."""
    import onnx.helper as h

    from optimize import prepare_for_quantization

    model = _make_preproc_conv_model()
    # Inject a stale value_info entry with a deliberately wrong dtype.
    model.graph.value_info.append(h.make_tensor_value_info("spec", onnx.TensorProto.INT64, [1, 4]))
    src = tmp_path / "src.onnx"
    onnx.save(model, str(src))
    prepared = tmp_path / "prepared.onnx"

    exclude = prepare_for_quantization(str(src), str(prepared))

    assert set(exclude) == {"pre_matmul", "pre_reshape"}
    assert prepared.exists()
    refreshed = onnx.load(str(prepared))
    onnx.checker.check_model(refreshed)
    vi = {v.name: v.type.tensor_type.elem_type for v in refreshed.graph.value_info}
    # The stale INT64 annotation must be replaced by the correct re-inferred FLOAT,
    # proving value_info was cleared AND re-inferred (not merely dropped).
    assert vi.get("spec") == onnx.TensorProto.FLOAT, "stale value_info must be re-inferred to FLOAT"


def test_int8_keeps_spectrogram_matmuls_full_precision(optimized_fp32_path: Path, tmp_path: Path):
    """INT8 dynamic quantization must leave the spectrogram MatMuls as float
    MatMul (not quantized) and clean up its temporary prepared model."""
    from optimize import find_preprocessing_node_names, quantize_to_int8_dynamic

    base = onnx.load(str(optimized_fp32_path))
    preprocessing = find_preprocessing_node_names(base)
    spectrogram_matmuls = {
        n.name for n in base.graph.node if n.op_type == "MatMul" and n.name in preprocessing
    }
    assert spectrogram_matmuls, "expected spectrogram MatMuls in the front-end"

    int8_path = tmp_path / "int8.onnx"
    assert quantize_to_int8_dynamic(str(optimized_fp32_path), str(int8_path))
    assert not (tmp_path / "int8.onnx.prep.onnx").exists(), "temp prepared model must be removed"

    quantized = onnx.load(str(int8_path))
    init_types = {i.name: i.data_type for i in quantized.graph.initializer}
    by_name = {n.name: n for n in quantized.graph.node}
    for name in spectrogram_matmuls:
        node = by_name[name]
        assert node.op_type == "MatMul", f"{name} was quantized (op={node.op_type})"
        assert any(init_types.get(i) == onnx.TensorProto.FLOAT for i in node.input), (
            f"{name} weights must remain FP32"
        )


def test_int8_arm_keeps_spectrogram_matmuls_full_precision(
    optimized_fp32_path: Path, tmp_path: Path
):
    """The ARM INT8 path (MatMul-only) must also leave the spectrogram MatMuls as
    float MatMul and clean up its temporary prepared model."""
    from optimize import find_preprocessing_node_names, quantize_to_int8_arm

    base = onnx.load(str(optimized_fp32_path))
    preprocessing = find_preprocessing_node_names(base)
    spectrogram_matmuls = {
        n.name for n in base.graph.node if n.op_type == "MatMul" and n.name in preprocessing
    }
    assert spectrogram_matmuls, "expected spectrogram MatMuls in the front-end"

    int8_path = tmp_path / "int8_arm.onnx"
    assert quantize_to_int8_arm(str(optimized_fp32_path), str(int8_path))
    assert not (tmp_path / "int8_arm.onnx.prep.onnx").exists(), (
        "temp prepared model must be removed"
    )

    quantized = onnx.load(str(int8_path))
    init_types = {i.name: i.data_type for i in quantized.graph.initializer}
    by_name = {n.name: n for n in quantized.graph.node}
    for name in spectrogram_matmuls:
        node = by_name[name]
        assert node.op_type == "MatMul", f"{name} was quantized (op={node.op_type})"
        assert any(init_types.get(i) == onnx.TensorProto.FLOAT for i in node.input), (
            f"{name} weights must remain FP32"
        )


def test_int8_warning_emitted_only_when_int8_produced():
    """The INT8 experimental warning must be emitted iff an INT8 variant is
    produced (including --int8-arm even with --no-int8), and the emitted text
    must name the accuracy risk and the opt-out so users do not ship a
    silently-degraded model."""
    from optimize import _print_int8_warning

    # (no_int8, int8_arm) -> whether an INT8 model is produced (so warn)
    expected = {
        (False, False): True,  # default: INT8 produced
        (True, False): False,  # --no-int8 only: nothing produced
        (True, True): True,  # --no-int8 --int8-arm: ARM still produced
        (False, True): True,  # --int8-arm: both produced
    }
    for (no_int8, int8_arm), should_warn in expected.items():
        captured: list[str] = []
        emitted = _print_int8_warning(no_int8, int8_arm, out=captured.append)
        assert emitted is should_warn, f"no_int8={no_int8} int8_arm={int8_arm}"
        assert bool(captured) is should_warn
        if should_warn:
            text = "\n".join(captured).lower()
            assert "experimental" in text and "accuracy" in text
            assert "--no-int8" in text and "fp16" in text


def _make_dft_mel_model(mel_weights):
    """A minimal runnable DFT->mel graph: X[1,4,8] @ Wdft[8,5] -> Relu -> @ Wmel[5,M].

    Wdft is a [F, F//2+1] cos/sin-style matrix in [-1, 1] (the DFT signature the
    truncation pass matches); Wmel is the [B, M] mel filterbank whose all-zero rows
    mark unused bins. M is kept != B//2+1 so the mel MatMul is not itself mistaken
    for a DFT.
    """
    import numpy as np
    import onnx.helper as h
    from onnx import TensorProto, numpy_helper

    fft, bins, frames = 8, 5, 4
    wdft = np.linspace(-1.0, 1.0, fft * bins, dtype=np.float32).reshape(fft, bins)
    wmel = np.asarray(mel_weights, dtype=np.float32)
    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, frames, fft])
    out = h.make_tensor_value_info("out", TensorProto.FLOAT, [1, frames, wmel.shape[1]])
    nodes = [
        h.make_node("MatMul", ["X", "wdft"], ["spec"], name="dft_matmul"),
        h.make_node("Relu", ["spec"], ["mag"], name="magnitude"),
        h.make_node("MatMul", ["mag", "wmel"], ["out"], name="mel_matmul"),
    ]
    inits = [
        numpy_helper.from_array(wdft, name="wdft"),
        numpy_helper.from_array(wmel, name="wmel"),
    ]
    graph = h.make_graph(nodes, "dft_mel", [x], [out], inits)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9
    return model


def test_truncate_dft_bins_drops_unused_bins_bit_exact():
    """DFT columns the mel filterbank zeroes out are removed (weight shrinks and a
    Pad restores the axis) while the graph output is numerically unchanged."""
    import numpy as np
    import onnxruntime as ort

    from optimize import truncate_dft_bins

    # mel filterbank [5, 2]: only bins 0,1,2 are used (rows 3,4 all zero)
    mel = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    model = _make_dft_mel_model(mel)
    original = model.SerializeToString()

    truncated, count = truncate_dft_bins(model)
    assert count == 1

    wdft = next(i for i in truncated.graph.initializer if i.name == "wdft")
    assert list(wdft.dims) == [8, 3], "DFT weight should be truncated 5 -> 3 columns"
    assert any(n.op_type == "Pad" for n in truncated.graph.node), "bin axis should be re-padded"

    x = np.random.default_rng(0).standard_normal((1, 4, 8)).astype(np.float32)
    base = ort.InferenceSession(original, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
    got = ort.InferenceSession(
        truncated.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"X": x})[0]
    # Not bit-identical: truncating the matmul changes its shape, so ORT may pick a
    # different kernel / reduction order, differing by float-reassociation noise
    # (~1e-6 on x86, exactly 0 on ARM in practice). A tolerance, not exact equality,
    # is the correct assertion; real corruption (a dropped used bin) is orders larger.
    assert np.abs(base - got).max() < 1e-4


def test_truncate_dft_bins_noop_when_all_bins_used():
    """A filterbank that touches every bin leaves the graph unchanged (no-op)."""
    from optimize import truncate_dft_bins

    mel = [[1.0, 0.0], [0.0, 1.0], [0.3, 0.2], [0.1, 0.4], [0.2, 0.1]]
    model = _make_dft_mel_model(mel)
    _, count = truncate_dft_bins(model)
    assert count == 0


def test_truncate_dft_bins_noop_without_dft():
    """A graph with no DFT-shaped MatMul is left untouched."""
    from optimize import truncate_dft_bins

    _, count = truncate_dft_bins(_make_preproc_conv_model())
    assert count == 0


def test_truncate_dft_bins_reverts_when_a_secondary_output_changes():
    """The bit-exact gate compares EVERY output, not just index 0. If a model also
    exposes the raw DFT bins as a separate output, truncation would change that
    output, so the pass must revert (protects multi-output models like --perch)."""
    import numpy as np
    import onnx.helper as h
    from onnx import TensorProto, numpy_helper

    from optimize import truncate_dft_bins

    fft, bins, frames = 8, 5, 4
    wdft = np.linspace(-1.0, 1.0, fft * bins, dtype=np.float32).reshape(fft, bins)
    wmel = np.asarray(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32
    )
    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, frames, fft])
    mel = h.make_tensor_value_info("mel", TensorProto.FLOAT, [1, frames, 2])
    spec = h.make_tensor_value_info("spec", TensorProto.FLOAT, [1, frames, bins])  # raw bins
    nodes = [
        h.make_node("MatMul", ["X", "wdft"], ["spec"], name="dft_matmul"),
        h.make_node("Relu", ["spec"], ["mag"], name="magnitude"),
        h.make_node("MatMul", ["mag", "wmel"], ["mel"], name="mel_matmul"),
    ]
    inits = [numpy_helper.from_array(wdft, name="wdft"), numpy_helper.from_array(wmel, name="wmel")]
    graph = h.make_graph(nodes, "dft_mel_multi", [x], [mel, spec], inits)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9

    before = model.SerializeToString()
    reverted, count = truncate_dft_bins(model)
    assert count == 0, "truncation must revert when a secondary output would change"
    assert reverted.SerializeToString() == before, "graph must be byte-identical after revert"


def test_create_dft_matrix_matches_exact_rfft():
    """The cosine basis must match the real part of an exact rfft to float32
    rounding. Building the phase in float32 drifts by up to ~6e-4 at BirdNET's
    FFT sizes, enough to move confidences away from the TFLite reference."""
    import numpy as np

    from optimize import create_dft_matrix

    for fft_size in (1024, 2048):  # BirdNET v2.4 high and low band
        matrix = create_dft_matrix(fft_size)
        expected = np.fft.rfft(np.eye(fft_size), axis=1).real
        assert matrix.dtype == np.float32
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-7)


def _make_framed_dft_mel_model(skew_frame=False, abs_between=False):
    """A BirdNET-shaped framed spectrogram front end on a [2, 64] waveform.

    X -> Reshape [-1, 32, 2] -> Gather(idx[9, 8], axis=1) -> Reshape [-1, 9, 16]
      -> Mul(hann) -> MatMul(dft[16, 9]) -> Reshape [-1, 9] -> MatMul(mel[9, 4])
      -> Reshape [-1, 9, 4] -> Mul(x, x)

    That is F=16 samples per frame, hop=6, gather group g=2, 9 frames. skew_frame
    shifts one frame index (non-contiguous framing); abs_between inserts an Abs
    between the DFT and mel MatMuls (a nonlinear magnitude, as in complex-STFT
    front ends).
    """
    import numpy as np
    import onnx.helper as h
    from onnx import TensorProto, numpy_helper

    from optimize import create_dft_matrix

    rng = np.random.default_rng(1)
    fft, bins, mels, frames, group, hop = 16, 9, 4, 9, 2, 6
    idx = np.arange(frames)[:, None] * (hop // group) + np.arange(fft // group)[None, :]
    if skew_frame:
        idx[4, 3] += 1
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fft) / fft)).astype(np.float32)
    mel = rng.random((bins, mels)).astype(np.float32)

    def i64(name, values):
        return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [2, 64])
    out = h.make_tensor_value_info("out", TensorProto.FLOAT, [2, frames, mels])
    inits = [
        i64("shape_grp", [-1, 64 // group, group]),
        i64("idx", idx),
        i64("shape_frames", [-1, frames, fft]),
        numpy_helper.from_array(hann, name="hann"),
        numpy_helper.from_array(create_dft_matrix(fft), name="dft"),
        i64("shape_flat", [-1, bins]),
        numpy_helper.from_array(mel, name="mel"),
        i64("shape_mel", [-1, frames, mels]),
    ]
    mel_in = "spec_flat"
    nodes = [
        h.make_node("Reshape", ["X", "shape_grp"], ["grp"], name="reshape_grp"),
        h.make_node("Gather", ["grp", "idx"], ["gathered"], name="gather_frames", axis=1),
        h.make_node("Reshape", ["gathered", "shape_frames"], ["frames"], name="reshape_frames"),
        h.make_node("Mul", ["frames", "hann"], ["windowed"], name="hann_mul"),
        h.make_node("MatMul", ["windowed", "dft"], ["spec"], name="dft_matmul"),
        h.make_node("Reshape", ["spec", "shape_flat"], ["spec_flat"], name="reshape_flat"),
    ]
    if abs_between:
        nodes.append(h.make_node("Abs", ["spec_flat"], ["spec_abs"], name="magnitude"))
        mel_in = "spec_abs"
    nodes += [
        h.make_node("MatMul", [mel_in, "mel"], ["mel_flat"], name="mel_matmul"),
        h.make_node("Reshape", ["mel_flat", "shape_mel"], ["mel_out"], name="reshape_mel"),
        h.make_node("Mul", ["mel_out", "mel_out"], ["out"], name="square"),
    ]
    graph = h.make_graph(nodes, "framed_dft_mel", [x], [out], inits)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def test_fold_frontend_to_conv_matches_framed_dft_mel():
    """The Hann/DFT/mel chain over framed samples becomes one strided 1-D Conv on
    the waveform with the same output (batch 2 pins the Unsqueeze/Transpose axes)."""
    import numpy as np
    import onnxruntime as ort

    from optimize import fold_frontend_to_conv

    model = _make_framed_dft_mel_model()
    original = model.SerializeToString()

    folded, count = fold_frontend_to_conv(model)
    assert count == 1
    ops = [n.op_type for n in folded.graph.node]
    assert "Gather" not in ops and "MatMul" not in ops
    convs = [n for n in folded.graph.node if n.op_type == "Conv"]
    assert len(convs) == 1
    weight = next(i for i in folded.graph.initializer if i.name == convs[0].input[1])
    assert list(weight.dims) == [4, 1, 16]
    assert list(next(a for a in convs[0].attribute if a.name == "strides").ints) == [6]

    x = np.random.default_rng(0).standard_normal((2, 64)).astype(np.float32)
    base = ort.InferenceSession(original, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
    got = ort.InferenceSession(folded.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        None, {"X": x}
    )[0]
    assert base.shape == got.shape == (2, 9, 4)
    assert np.abs(base - got).max() < 1e-4


@pytest.mark.parametrize(
    "mutation", [{"skew_frame": True}, {"abs_between": True}], ids=["skewed_frame", "nonlinear"]
)
def test_fold_frontend_to_conv_skips_unfoldable(mutation):
    """Non-contiguous framing or a nonlinearity between the DFT and mel MatMuls is
    not a linear strided filter, so nothing is folded and the graph is untouched."""
    from optimize import fold_frontend_to_conv

    model = _make_framed_dft_mel_model(**mutation)
    before = model.SerializeToString()
    result, count = fold_frontend_to_conv(model)
    assert count == 0
    assert result.SerializeToString() == before


def test_optimize_model_fails_when_fold_requested_but_not_applied(tmp_path: Path):
    """A requested fold that cannot be applied fails the run instead of falling
    back to truncation, so a truncated graph is never mistaken for a folded one."""
    from optimize import optimize_model

    mel = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    src = tmp_path / "toy.onnx"
    onnx.save(_make_dft_mel_model(mel), str(src))
    assert optimize_model(str(src), fold_frontend=True) is None


def test_fold_frontend_matches_default_on_real_audio(
    optimized_fp32_path: Path, optimized_fold_fp32_path: Path, audio_windows
):
    """On real audio the folded build agrees with the default build (the only
    difference between them is the fold): same top-1, logits within 5e-3."""
    import numpy as np
    import onnxruntime as ort

    folded = onnx.load(str(optimized_fold_fp32_path))
    init_rank = {i.name: len(i.dims) for i in folded.graph.initializer}
    convs = [n for n in folded.graph.node if n.op_type == "Conv"]
    assert [init_rank[c.input[1]] for c in convs[:2]] == [3, 3], "both bands fold to 1-D Convs"
    ops = [n.op_type for n in folded.graph.node]
    assert "Gather" not in ops
    assert ops.count("MatMul") == 1, "only the classifier head MatMul should remain"

    def logits(path: Path) -> np.ndarray:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        return np.concatenate(
            [
                session.run(None, {name: audio_windows[i : i + 16]})[0]
                for i in range(0, len(audio_windows), 16)
            ]
        )

    ref = logits(optimized_fp32_path)
    got = logits(optimized_fold_fp32_path)
    assert ref.shape == got.shape == (len(audio_windows), 6522)
    assert np.abs(ref - got).max() <= 5e-3
    assert (ref.argmax(axis=1) == got.argmax(axis=1)).all()
    top5_ref = np.sort(np.argsort(-ref, axis=1)[:, :5], axis=1)
    top5_got = np.sort(np.argsort(-got, axis=1)[:, :5], axis=1)
    assert (top5_ref == top5_got).all(axis=1).mean() >= 0.99


def test_fold_frontend_fp16_keeps_fold_conv_fp32(optimized_fold_fp32_path: Path):
    """The folded front-end Convs and the Pow nodes are protected, so
    convert_to_fp16 keeps the fold weights FP32 (FP16 with --fp16-full) and the
    protected model still runs in ONNX Runtime."""
    import numpy as np
    import onnxruntime as ort

    from optimize import convert_to_fp16, find_preprocessing_node_names

    base = onnx.load(str(optimized_fold_fp32_path))
    init_rank = {i.name: len(i.dims) for i in base.graph.initializer}
    fold_convs = [n for n in base.graph.node if n.op_type == "Conv" and init_rank[n.input[1]] == 3]
    assert len(fold_convs) == 2
    fold_weights = {n.input[1] for n in fold_convs}
    preprocessing = find_preprocessing_node_names(base)
    assert {n.name for n in fold_convs} <= preprocessing
    assert {n.name for n in base.graph.node if n.op_type == "Pow"} <= preprocessing

    def weight_dtypes(model: onnx.ModelProto) -> set[int]:
        return {i.data_type for i in model.graph.initializer if i.name in fold_weights}

    protected = convert_to_fp16(onnx.load(str(optimized_fold_fp32_path)))
    full = convert_to_fp16(onnx.load(str(optimized_fold_fp32_path)), keep_preprocessing_fp32=False)
    assert protected is not None and full is not None
    assert weight_dtypes(protected) == {onnx.TensorProto.FLOAT}
    assert weight_dtypes(full) == {onnx.TensorProto.FLOAT16}

    session = ort.InferenceSession(
        protected.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    spec = session.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
    session.run(None, {spec.name: np.zeros(shape, dtype=np.float32)})
