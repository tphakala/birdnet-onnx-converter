#!/bin/bash
# Setup script for BirdNET ONNX Converter
# Creates a virtual environment and installs all dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== BirdNET ONNX Converter Setup ==="
echo

# Check Python version
PYTHON_CMD=""
for cmd in python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.9 or higher is required"
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo
echo "Installing core dependencies..."
pip install onnx onnxslim onnxscript numpy onnxruntime onnxconverter-common --quiet

echo "Installing TFLite conversion dependencies..."
pip install tensorflow tf2onnx --quiet

# Optional: onnx-simplifier (may fail on some systems)
echo "Attempting to install onnx-simplifier (optional)..."
if pip install onnx-simplifier --quiet 2>/dev/null; then
    echo "  onnx-simplifier installed"
else
    echo "  onnx-simplifier skipped (requires cmake)"
fi

echo
echo "Setting up git hooks..."
git config core.hooksPath .githooks 2>/dev/null && echo "  Git hooks enabled" || echo "  Git hooks skipped (not in a git repo)"

echo
echo "=== Setup Complete ==="
echo
echo "To use the converter:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Then run:"
echo "  python convert.py --input model.tflite --output-dir ./"
echo "  python optimize.py --input model.onnx --output model"
echo
echo "To deactivate the virtual environment:"
echo "  deactivate"
