"""Tests for ONNX model inference."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

# BirdNET v2.4 has 6522 species
EXPECTED_SPECIES_COUNT = 6522

# 3 seconds of audio at 48kHz
SAMPLE_RATE = 48000
DURATION_SECONDS = 3
INPUT_LENGTH = SAMPLE_RATE * DURATION_SECONDS


def test_fp32_inference(optimized_fp32_path: Path):
    """Verify FP32 model runs inference correctly."""
    session = ort.InferenceSession(
        str(optimized_fp32_path),
        providers=["CPUExecutionProvider"],
    )

    # Create dummy input (3 seconds of silence at 48kHz)
    dummy_input = np.zeros((1, INPUT_LENGTH), dtype=np.float32)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Run inference
    output = session.run(None, {input_name: dummy_input})

    # Verify output shape
    assert len(output) == 1, "Model should have one output"
    assert output[0].shape == (1, EXPECTED_SPECIES_COUNT), (
        f"Output shape should be (1, {EXPECTED_SPECIES_COUNT}), got {output[0].shape}"
    )


def test_fp16_inference(optimized_fp16_path: Path):
    """Verify FP16 model runs inference correctly."""
    session = ort.InferenceSession(
        str(optimized_fp16_path),
        providers=["CPUExecutionProvider"],
    )

    # Create dummy input (FP16 models still take FP32 input with keep_io_types=True)
    dummy_input = np.zeros((1, INPUT_LENGTH), dtype=np.float32)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Run inference
    output = session.run(None, {input_name: dummy_input})

    # Verify output shape
    assert len(output) == 1, "Model should have one output"
    assert output[0].shape == (1, EXPECTED_SPECIES_COUNT), (
        f"Output shape should be (1, {EXPECTED_SPECIES_COUNT}), got {output[0].shape}"
    )


def test_fp32_output_has_valid_shape(optimized_fp32_path: Path):
    """Verify FP32 model output has expected dimensions."""
    session = ort.InferenceSession(
        str(optimized_fp32_path),
        providers=["CPUExecutionProvider"],
    )

    dummy_input = np.zeros((1, INPUT_LENGTH), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy_input})[0]

    # Model outputs logits (raw scores), not probabilities
    # Verify output is finite and has reasonable range for logits
    assert np.all(np.isfinite(output)), "Output should be finite"
    assert output.shape == (1, EXPECTED_SPECIES_COUNT)


def test_fp16_output_has_valid_shape(optimized_fp16_path: Path):
    """Verify FP16 model output has expected dimensions."""
    session = ort.InferenceSession(
        str(optimized_fp16_path),
        providers=["CPUExecutionProvider"],
    )

    dummy_input = np.zeros((1, INPUT_LENGTH), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy_input})[0]

    # Model outputs logits (raw scores), not probabilities
    # Verify output is finite and has reasonable range for logits
    assert np.all(np.isfinite(output)), "Output should be finite"
    assert output.shape == (1, EXPECTED_SPECIES_COUNT)


def _resolve_output(session: ort.InferenceSession, outputs: list, name_contains: str):
    """Look up a model output by name substring, falling back to index 0.

    Output ordering is not guaranteed across converters/formats, so multi-output
    consumers should select by name rather than position.
    """
    for i, meta in enumerate(session.get_outputs()):
        if name_contains.lower() in meta.name.lower():
            return outputs[i]
    return outputs[0]


def test_output_resolution_is_name_based_not_positional():
    """A two-output model is queried by name, so the correct tensor is returned
    regardless of output order."""
    import onnx.helper as h
    from onnx import TensorProto

    x = h.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    emb = h.make_tensor_value_info("embeddings", TensorProto.FLOAT, [1, 2])
    pred = h.make_tensor_value_info("predictions", TensorProto.FLOAT, [1, 3])
    we = h.make_tensor("we", TensorProto.FLOAT, [2, 2], [1, 0, 0, 1])
    wp = h.make_tensor("wp", TensorProto.FLOAT, [2, 3], [1, 0, 0, 0, 1, 0])
    nodes = [
        h.make_node("MatMul", ["X", "we"], ["embeddings"]),
        h.make_node("MatMul", ["X", "wp"], ["predictions"]),
    ]
    # Outputs deliberately ordered embeddings-first to defeat index-0 assumptions.
    graph = h.make_graph(nodes, "two_out", [x], [emb, pred], [we, wp])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 9

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"X": np.ones((1, 2), dtype=np.float32)})

    predictions = _resolve_output(session, outputs, "predictions")
    assert predictions.shape == (1, 3), "name-based lookup must return the predictions tensor"
    # Positional access would have returned embeddings (shape (1, 2)) here.
    assert session.get_outputs()[0].name == "embeddings"


def test_fp32_inference_batch_sweep(optimized_fp32_path: Path):
    """The dynamic batch axis works across batch sizes (the variable axis for the
    fixed-window v2.4 model); output batch dim must track the input."""
    session = ort.InferenceSession(str(optimized_fp32_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    for batch in (1, 2, 3):
        dummy = np.zeros((batch, INPUT_LENGTH), dtype=np.float32)
        output = _resolve_output(session, session.run(None, {input_name: dummy}), "output")
        assert output.shape == (batch, EXPECTED_SPECIES_COUNT), (
            f"batch {batch} should yield {(batch, EXPECTED_SPECIES_COUNT)}, got {output.shape}"
        )


def test_fp16_inference_batch_sweep(optimized_fp16_path: Path):
    """The FP16 model's dynamic batch axis also works across batch sizes (the
    FP16 path has its own keep_io_types boundary, so it is worth covering)."""
    session = ort.InferenceSession(str(optimized_fp16_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    for batch in (1, 2):
        dummy = np.zeros((batch, INPUT_LENGTH), dtype=np.float32)
        output = _resolve_output(session, session.run(None, {input_name: dummy}), "output")
        assert output.shape == (batch, EXPECTED_SPECIES_COUNT)


def test_fp32_inference_on_audio(optimized_fp32_path: Path, audio_input):
    """The model runs on (deterministic) audio input and produces finite logits."""
    session = ort.InferenceSession(str(optimized_fp32_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = _resolve_output(session, session.run(None, {input_name: audio_input}), "output")
    assert output.shape == (1, EXPECTED_SPECIES_COUNT)
    assert np.all(np.isfinite(output))


def test_labels_match_output_size(optimized_fp32_path: Path, labels_path: Path):
    """Verify labels file has same count as model output."""
    session = ort.InferenceSession(
        str(optimized_fp32_path),
        providers=["CPUExecutionProvider"],
    )

    dummy_input = np.zeros((1, INPUT_LENGTH), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy_input})[0]

    # Count labels
    with open(labels_path) as f:
        label_count = sum(1 for line in f if line.strip())

    assert label_count == output.shape[1], (
        f"Labels count ({label_count}) should match output size ({output.shape[1]})"
    )
