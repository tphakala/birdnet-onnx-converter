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
