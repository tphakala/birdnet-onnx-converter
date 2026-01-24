"""Tests for TFLite to ONNX conversion."""

from pathlib import Path

import onnx


def test_tflite_to_onnx_conversion(tflite_model_path: Path, tmp_path: Path):
    """Verify TFLite model converts to ONNX."""
    from convert import convert_tflite_to_onnx

    output_path = tmp_path / "model.onnx"
    convert_tflite_to_onnx(str(tflite_model_path), str(output_path))

    assert output_path.exists(), "ONNX file was not created"

    # Load model (skip onnx.checker - raw model has custom ops like RFFT2D)
    model = onnx.load(str(output_path))
    assert len(model.graph.node) > 0, "Model should have nodes"


def test_converted_model_has_expected_io(converted_onnx_path: Path):
    """Verify converted model has expected input/output structure."""
    model = onnx.load(str(converted_onnx_path))

    # Check inputs
    assert len(model.graph.input) >= 1, "Model should have at least one input"

    # Check outputs
    assert len(model.graph.output) >= 1, "Model should have at least one output"


def test_converted_model_has_nodes(converted_onnx_path: Path):
    """Verify converted model has expected structure."""
    model = onnx.load(str(converted_onnx_path))
    # Skip onnx.checker - raw model has custom ops (RFFT2D) that need optimization
    assert len(model.graph.node) > 100, "Model should have many nodes"
