# BirdNET ONNX Converter

Convert and optimize BirdNET models for ONNX Runtime inference on various platforms including GPUs, desktop CPUs, and embedded devices like Raspberry Pi.

## Features

- **TFLite to ONNX conversion** - Convert BirdNET TFLite models to ONNX format
- **GPU optimization** - Replace unsupported ops (RFFT2D, ReverseSequence) with GPU-friendly alternatives
- **Multiple precision formats**:
  - FP32 - Standard precision for GPU/desktop
  - FP16 - Half precision for devices with FP16 support (RPi 5, modern GPUs)

## Installation

### Quick Setup (Recommended)

Use the setup script to create an isolated virtual environment:

```bash
./setup.sh
source .venv/bin/activate
```

### Manual Installation

If you prefer to manage dependencies yourself:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install -r requirements.txt

# For TFLite conversion (optional)
pip install -r requirements-tflite.txt

# Optional: onnx-simplifier (requires cmake)
pip install onnx-simplifier
```

## Usage

### Step 1: Convert TFLite to ONNX

```bash
python convert.py --input BirdNET_Model.tflite --output-dir ./ --onnx-only
```

### Step 2: Optimize ONNX Model

```bash
# Output all formats (FP32, FP16)
python optimize.py --input BirdNET_Model.onnx --output BirdNET

# Output only FP32
python optimize.py --input BirdNET_Model.onnx --output BirdNET --fp32-only
```

### Output Files

| File | Description | Recommended Use |
| ---- | ----------- | --------------- |
| `*_fp32.onnx` | Full precision | GPU (CUDA/TensorRT), Desktop CPU |
| `*_fp16.onnx` | Half precision | RPi 5, Modern GPUs |

## Key Optimizations

The optimizer applies the following transformations:

1. **RFFT2D → MatMul** - Replaces TensorFlow's RFFT2D with precomputed DFT matrix multiplication
2. **ReverseSequence → Slice** - Replaces with negative-stride slice operation
3. **GlobalAveragePool + Squeeze → ReduceMean** - Combines into single operation
4. **Transpose fusion** - Merges consecutive transpose operations
5. **Cast removal** - Removes all Cast nodes via onnxscript rewriter for proper type propagation
6. **INT32 → INT64 conversion** - Converts integer initializers to INT64 for better compatibility
7. **Graph optimization** - Uses onnxscript optimizer and onnxslim for further optimization
8. **Split node fixing** - Removes zero-size outputs from Split nodes
9. **Dead code elimination** - Removes orphaned nodes not contributing to output

## Supported Models

This tool supports conversion and optimization of:

- **Official BirdNET models** - Download from [Zenodo](https://zenodo.org/records/15050749) (v2.4 with 6522 species)
- **Custom classifiers** - Models trained with [BirdNET Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) for additional species or regional variants
- **Any BirdNET variant** - Models using the same backbone architecture

## Platform-Specific Notes

### Raspberry Pi 5

Use FP16 model for best performance:

```python
import onnxruntime as ort
session = ort.InferenceSession("BirdNET_fp16.onnx")
```

### Raspberry Pi 3/4

Use FP32 model (FP16 not natively supported):

```python
session = ort.InferenceSession("BirdNET_fp32.onnx")
```

### NVIDIA GPUs

Use FP32 or FP16 with CUDA/TensorRT:

```python
session = ort.InferenceSession(
    "BirdNET_fp16.onnx",
    providers=['CUDAExecutionProvider']
)
```

## Acknowledgments

This project is based on [Justin Chu's optimized ONNX conversion](https://huggingface.co/justinchuby/BirdNET-onnx) of the BirdNET v2.4 model. His work on replacing TensorFlow-specific operations with ONNX-compatible alternatives made GPU inference possible.

## References

- [BirdNET Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) - Official BirdNET tool for training custom classifiers
- [BirdNET Models on Zenodo](https://zenodo.org/records/15050749) - Official model downloads
- [Justin Chu's BirdNET-ONNX](https://huggingface.co/justinchuby/BirdNET-onnx) - Original ONNX conversion work
- [ONNX Runtime](https://onnxruntime.ai/)

## License

MIT License
