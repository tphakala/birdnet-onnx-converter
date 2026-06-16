# BirdNET ONNX Converter

Convert and optimize BirdNET models for ONNX Runtime inference on various platforms including GPUs, desktop CPUs, and embedded devices like Raspberry Pi.

## Features

- **Multiple input formats** - Convert from Keras (.keras), TFLite (.tflite), or TensorFlow SavedModel directories
- **GPU optimization** - Replace unsupported ops (RFFT2D, ReverseSequence, DFT) with GPU-friendly alternatives
- **Multi-output model support** - Handle models like Google Perch v2 with multiple outputs
- **Output pruning** - Strip unused model outputs to reduce compute
- **Multiple precision formats**:
  - FP32 - Standard precision for GPU/desktop
  - FP16 - Half precision for devices with FP16 support (RPi 5, modern GPUs). The
    spectrogram preprocessing front-end (STFT/mel MatMuls and normalization) is
    kept in FP32 by default to preserve accuracy; pass `--fp16-full` to convert
    the entire graph. On ARM/CPU, `--fp16-keep-activations-fp32` keeps the
    Swish/SiLU activations in FP32 so ONNX Runtime's CPU EP can fuse QuickGelu,
    recovering most of the FP16 latency (weights stay FP16; small runtime-RAM cost).
  - INT8 - Dynamic quantization for low-power CPUs (experimental; validate
    accuracy before use). The spectrogram preprocessing is excluded so it stays
    full precision. `--int8-arm` quantizes only MatMul ops (skipping Conv) to
    avoid ConvInteger kernels that ARM ONNX Runtime builds do not support.

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

### Step 1: Convert to ONNX

```bash
# From TFLite
python convert.py --input BirdNET_Model.tflite --output-dir ./ --onnx-only

# From TensorFlow SavedModel directory
python convert.py --input /path/to/saved_model/ --output-dir ./ --name my_model
```

Note: JAX-based SavedModels (like Google Perch v2) contain `XlaCallModule` ops that
tf2onnx cannot decompose. For those models, convert from TFLite instead, or use a
pre-converted ONNX model and run `optimize.py` directly.

### Step 2: Optimize ONNX Model

```bash
# BirdNET: output all formats (FP32, FP16, INT8)
python optimize.py --input BirdNET_Model.onnx --output BirdNET

# BirdNET: output only FP32
python optimize.py --input BirdNET_Model.onnx --output BirdNET --fp32-only

# BirdNET: skip INT8 (FP32 + FP16 only)
python optimize.py --input BirdNET_Model.onnx --output BirdNET --no-int8

# BirdNET: also produce an ARM-friendly INT8 model (MatMul-only)
python optimize.py --input BirdNET_Model.onnx --output BirdNET --int8-arm

# FP16: convert the whole graph, including spectrogram preprocessing
# (by default the preprocessing front-end is kept in FP32 for accuracy)
python optimize.py --input BirdNET_Model.onnx --output BirdNET --fp16-full

# FP16: keep the Swish/SiLU activations (Sigmoid x Mul) in FP32 so ONNX Runtime's
# CPU EP can fuse QuickGelu (much faster FP16 on ARM/CPU; weights stay FP16)
python optimize.py --input BirdNET_Model.onnx --output BirdNET --fp16-keep-activations-fp32

# Perch v2: preserve multi-output naming, include ARM-optimized INT8
python optimize.py --input perch_v2.onnx --output perch_v2 --perch --int8-arm

# Perch v2: prune unused outputs (keep only embeddings and labels)
python optimize.py --input perch_v2.onnx --output perch_v2_lite --perch \
    --prune-outputs embedding,label

# Range filter / SelectV2 fix: collapse a tf2onnx where() Cast/Mul/Add cluster
# into a single Where op (no other optimization). Use for models arm64 ONNX
# Runtime rejects due to the bool->float Cast in the decomposition (e.g. the
# BirdNET MData range filter), which the full optimize pipeline would corrupt.
python optimize.py --input mdata_raw.onnx --output mdata_where.onnx \
    --collapse-select-to-where
```

### Output Files

| File | Description | Recommended Use |
| ---- | ----------- | --------------- |
| `*_fp32.onnx` | Full precision | GPU (CUDA/TensorRT), Desktop CPU |
| `*_fp16.onnx` | Half precision | RPi 5, Modern GPUs |
| `*_int8.onnx` | INT8 dynamic quantization (experimental) | Low-power x86 CPU |
| `*_int8_arm.onnx` | INT8, MatMul only, `--int8-arm` (experimental) | RPi 3/4, ARM CPU |

For both FP16 and INT8, the spectrogram preprocessing front-end (STFT/mel
MatMuls and normalization) is kept in full precision to preserve accuracy;
only the CNN backbone is reduced.

> **INT8 is experimental and not recommended for production.** Dynamic
> quantization can significantly degrade accuracy and may change the top
> predicted species. Always validate INT8 output against FP32/FP16, and prefer
> FP16 where the target supports it. Pass `--no-int8` (and omit `--int8-arm`) to
> skip INT8.

## Key Optimizations

The optimizer applies the following transformations:

1. **RFFT2D → MatMul** - Replaces TensorFlow's RFFT2D with precomputed DFT matrix multiplication
2. **ReverseSequence → Slice** - Replaces with negative-stride slice operation
3. **GlobalAveragePool + Squeeze → ReduceMean** - Combines into single operation
4. **Transpose fusion** - Merges consecutive transpose operations
5. **Cast removal** - Removes redundant/identity Cast nodes for proper type propagation
6. **INT32 → INT64 conversion** - Converts integer initializers to INT64 for better compatibility
7. **Graph optimization** - Uses onnxscript optimizer and onnxslim for further optimization
8. **Split node fixing** - Removes zero-size outputs from Split nodes
9. **Dead code elimination** - Removes orphaned nodes not contributing to output

### Precision conversion safeguards

When emitting reduced-precision variants, the converter protects the
accuracy-critical spectrogram front-end:

- **FP16 dual-role output decoupling** - Inserts an Identity boundary so the
  `keep_io_types` Cast cannot leave mixed fp16/fp32 ops on multi-output models
  (e.g. the v2.4 embeddings model); external output names are preserved.
- **FP16 preprocessing protection** - Keeps the STFT/mel MatMuls and
  normalization in FP32 (the CNN backbone goes FP16); `--fp16-full` opts out.
- **INT8 preprocessing exclusion** - Excludes the spectrogram MatMuls from INT8
  quantization and refreshes `value_info` before quantizing so tensor types are
  classified correctly.

## Supported Models

This tool supports conversion and optimization of:

- **Official BirdNET models** - Download from [Zenodo](https://zenodo.org/records/15050749) (v2.4 with 6522 species)
- **Google Perch v2** - Bird vocalization classifier with ~14,795 species ([Kaggle](https://www.kaggle.com/models/google/bird-vocalization-classifier/))
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
