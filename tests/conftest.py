"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def output_dir() -> Path:
    """Return path to test output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture(scope="session")
def tflite_model_path(fixtures_dir: Path) -> Path:
    """Return path to TFLite model fixture."""
    path = fixtures_dir / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
    if not path.exists():
        pytest.skip(f"TFLite model not found: {path}")
    return path


@pytest.fixture(scope="session")
def labels_path(fixtures_dir: Path) -> Path:
    """Return path to labels file fixture."""
    path = fixtures_dir / "BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt"
    if not path.exists():
        pytest.skip(f"Labels file not found: {path}")
    return path


@pytest.fixture(scope="session")
def test_audio_path(fixtures_dir: Path) -> Path:
    """Return path to test audio fixture."""
    path = fixtures_dir / "test_audio.wav"
    if not path.exists():
        pytest.skip(f"Test audio not found: {path}")
    return path


@pytest.fixture(scope="session")
def audio_input(fixtures_dir: Path):
    """Return a deterministic (1, 144000) float32 BirdNET input.

    Uses the real test audio when present, otherwise falls back to seeded noise
    so inference tests stay runnable and reproducible without the WAV asset.

    Assumes 16-bit PCM at 48 kHz and does not resample (it pads/truncates to
    144000 samples = 3 s). The array is session-scoped and shared; consumers must
    not mutate it in place.
    """
    import wave

    import numpy as np

    rate = 48000
    samples = rate * 3  # BirdNET v2.4: 3 s at 48 kHz
    path = fixtures_dir / "test_audio.wav"
    if path.exists():
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            assert w.getsampwidth() == 2, "test audio must be 16-bit PCM"
            assert w.getframerate() == rate, f"test audio must be {rate} Hz (no resampling)"
            # WAV/RIFF is little-endian; '<i2' decodes correctly on any host.
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        audio = raw.astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
    else:
        audio = np.random.default_rng(42).standard_normal(samples).astype(np.float32) * 0.1
    audio = np.pad(audio, (0, max(0, samples - len(audio))))[:samples]
    return audio.reshape(1, samples).astype(np.float32)


@pytest.fixture(scope="session")
def baseline_predictions_path(fixtures_dir: Path) -> Path:
    """Return path to baseline predictions fixture."""
    path = fixtures_dir / "baseline_predictions.csv"
    if not path.exists():
        pytest.skip(f"Baseline predictions not found: {path}")
    return path


@pytest.fixture(scope="session")
def converted_onnx_path(tflite_model_path: Path, output_dir: Path) -> Path:
    """Convert TFLite to ONNX and return path."""
    from convert import convert_tflite_to_onnx

    output_path = output_dir / "BirdNET_converted.onnx"
    if not output_path.exists():
        convert_tflite_to_onnx(str(tflite_model_path), str(output_path))
    return output_path


@pytest.fixture(scope="session")
def optimized_fp32_path(converted_onnx_path: Path, output_dir: Path) -> Path:
    """Optimize ONNX model and return FP32 path."""
    import onnx

    from optimize import optimize_model

    output_path = output_dir / "BirdNET_fp32.onnx"
    if not output_path.exists():
        optimized = optimize_model(str(converted_onnx_path))
        assert optimized is not None, "Model optimization failed"
        onnx.save(optimized, str(output_path))
    return output_path


@pytest.fixture(scope="session")
def optimized_fp16_path(optimized_fp32_path: Path, output_dir: Path) -> Path:
    """Convert optimized model to FP16 and return path."""
    import onnx

    from optimize import convert_to_fp16

    output_path = output_dir / "BirdNET_fp16.onnx"
    if not output_path.exists():
        model = onnx.load(str(optimized_fp32_path))
        fp16_model = convert_to_fp16(model)
        if fp16_model is not None:
            onnx.save(fp16_model, str(output_path))
    if not output_path.exists():
        pytest.skip("FP16 conversion failed")
    return output_path
