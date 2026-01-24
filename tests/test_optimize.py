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
