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


def test_find_preprocessing_node_names_uses_first_conv():
    """With multiple Convs, only ancestors of the FIRST Conv are the front-end;
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

    # The Swish Sigmoid output runs in FP32.
    vi = {v.name: v.type.tensor_type.elem_type for v in kept.graph.value_info}
    sig = next(n for n in kept.graph.node if n.op_type == "Sigmoid")
    assert vi.get(sig.output[0]) == onnx.TensorProto.FLOAT, "Sigmoid must stay FP32"


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
