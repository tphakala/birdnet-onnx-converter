#!/usr/bin/env python3
"""
Convert Keras model to TFLite and ONNX formats.

This script handles conversion of BirdNET Keras models (.keras format) to
TFLite and ONNX formats for deployment on various platforms.

Usage:
    python convert.py [--input model.keras] [--output-dir ./]

Requirements:
    pip install tensorflow tf2onnx onnx
"""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path


def load_keras_model_manual(keras_path: str):
    """
    Load a .keras model by manually extracting and rebuilding it.
    This handles version incompatibility issues between Keras versions.
    """
    import h5py
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract the .keras archive
        with zipfile.ZipFile(keras_path, "r") as z:
            z.extractall(tmpdir)

        # Load config
        config_path = Path(tmpdir) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Parse architecture from config
        model_config = config["config"]
        layers_config = model_config["layers"]

        # Rebuild model from scratch based on the architecture
        inputs = None
        x = None

        for layer_cfg in layers_config:
            class_name = layer_cfg["class_name"]
            layer_config = layer_cfg["config"]

            if class_name == "InputLayer":
                batch_shape = layer_config.get("batch_input_shape", layer_config.get("batch_shape"))
                inputs = tf.keras.Input(shape=batch_shape[1:], name=layer_config["name"])
                x = inputs
            elif class_name == "Dense":
                layer = tf.keras.layers.Dense(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    use_bias=layer_config["use_bias"],
                    name=layer_config["name"],
                )
                x = layer(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)

        # Load weights from h5 file
        weights_path = Path(tmpdir) / "model.weights.h5"
        if weights_path.exists():
            with h5py.File(weights_path, "r") as f:
                # Navigate the h5 file structure to load weights
                for layer in model.layers:
                    if layer.name in f:
                        layer_group = f[layer.name]
                        if layer.name in layer_group:
                            vars_group = layer_group[layer.name]
                            weights = []
                            # Load kernel and bias in order
                            if "kernel:0" in vars_group:
                                weights.append(vars_group["kernel:0"][:])
                            if "bias:0" in vars_group:
                                weights.append(vars_group["bias:0"][:])
                            if weights:
                                layer.set_weights(weights)

        return model


def convert_to_tflite(model, output_path: str, optimize: bool = True) -> str:
    """Convert Keras model to TFLite format."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if optimize:
        # Apply default optimizations (quantization-aware)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return output_path


def convert_to_onnx(model, output_path: str, opset: int = 13) -> str:
    """Convert Keras model to ONNX format."""
    import tensorflow as tf
    import tf2onnx

    # Get input spec from model
    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input")]

    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=output_path,
    )

    return output_path


def convert_tflite_to_onnx(tflite_path: str, output_path: str, opset: int = 17) -> str:
    """Convert TFLite model to ONNX format using tf2onnx CLI."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--tflite",
        tflite_path,
        "--output",
        output_path,
        "--opset",
        str(opset),
        "--continue_on_error",  # Handle numpy 2.0 compatibility
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"tf2onnx conversion failed with exit code {result.returncode}:\n{result.stderr}"
        )

    return output_path


def verify_tflite(tflite_path: str, input_shape: tuple) -> bool:
    """Verify TFLite model loads and runs."""
    import numpy as np
    import tensorflow as tf

    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test with random input
        test_input = np.random.randn(1, *input_shape[1:]).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


def verify_onnx(onnx_path: str, input_shape: tuple) -> bool:
    """Verify ONNX model loads and is valid."""
    import numpy as np
    import onnx

    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Try inference with onnxruntime if available
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            test_input = np.random.randn(1, *input_shape[1:]).astype(np.float32)
            output = session.run(None, {input_name: test_input})
            print(f"  Input name: {input_name}")
            print(f"  Output shape: {output[0].shape}")
        except ImportError:
            print("  onnxruntime not installed, skipping inference test")

        return True
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Keras/TFLite model to TFLite and ONNX formats"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input model path (.keras or .tflite)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Output directory for converted models (default: current directory)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable TFLite optimizations",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip model verification after conversion",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Only convert to ONNX (for TFLite input)",
    )
    parser.add_argument(
        "--unsafe-fallback",
        action="store_true",
        help="Allow fallback to tf.keras.models.load_model (may execute arbitrary code)",
    )
    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input model not found: {input_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output paths
    model_name = input_path.stem
    tflite_path = output_dir / f"{model_name}.tflite"
    onnx_path = output_dir / f"{model_name}.onnx"

    # Handle TFLite input
    if input_path.suffix == ".tflite":
        print(f"Input is TFLite model: {input_path}")

        if args.onnx_only:
            print(f"\nConverting to ONNX: {onnx_path}")
            try:
                convert_tflite_to_onnx(str(input_path), str(onnx_path), opset=args.opset)
                onnx_size = onnx_path.stat().st_size / (1024 * 1024)
                print(f"  ONNX model saved ({onnx_size:.2f} MB)")
            except Exception as e:
                print(f"  ONNX conversion failed: {e}")
                return 1
        else:
            print("TFLite to TFLite conversion not needed")
            print("Use --onnx-only to convert TFLite to ONNX")
            return 0

        print("\nConversion complete!")
        print(f"  ONNX: {onnx_path}")
        return 0

    # Handle Keras input
    print(f"Loading Keras model: {input_path}")
    try:
        model = load_keras_model_manual(str(input_path))
        print("  Model loaded successfully (manual extraction)")
    except Exception as e:
        print(f"  Manual loading failed: {e}")
        if not args.unsafe_fallback:
            print("  Error: Safe model loading failed.")
            print("  The fallback method (tf.keras.models.load_model) can execute")
            print("  arbitrary code from untrusted model files.")
            print("  Use --unsafe-fallback to allow this fallback method.")
            return 1
        print("  WARNING: Using unsafe fallback (tf.keras.models.load_model)")
        print("  This may execute arbitrary code from the model file!")
        import tensorflow as tf

        model = tf.keras.models.load_model(input_path)

    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # Convert to TFLite
    print(f"\nConverting to TFLite: {tflite_path}")
    try:
        convert_to_tflite(model, str(tflite_path), optimize=not args.no_optimize)
        tflite_size = tflite_path.stat().st_size / (1024 * 1024)
        print(f"  TFLite model saved ({tflite_size:.2f} MB)")

        if not args.skip_verify:
            print("  Verifying TFLite model...")
            verify_tflite(str(tflite_path), model.input_shape)
    except Exception as e:
        print(f"  TFLite conversion failed: {e}")
        return 1

    # Convert to ONNX
    print(f"\nConverting to ONNX: {onnx_path}")
    try:
        convert_to_onnx(model, str(onnx_path), opset=args.opset)
        onnx_size = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX model saved ({onnx_size:.2f} MB)")

        if not args.skip_verify:
            print("  Verifying ONNX model...")
            verify_onnx(str(onnx_path), model.input_shape)
    except Exception as e:
        print(f"  ONNX conversion failed: {e}")
        return 1

    print("\nConversion complete!")
    print(f"  TFLite: {tflite_path}")
    print(f"  ONNX: {onnx_path}")
    return 0


if __name__ == "__main__":
    exit(main())
