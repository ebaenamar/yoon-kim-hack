#!/bin/bash

echo "=========================================="
echo "Setting up with Python 3.11 for FLA"
echo "=========================================="

cd /home/nuwins-server1/CascadeProjects/architecture-benchmark

# Use Python 3.11
PYTHON=/usr/bin/python3.11

echo "Python version:"
$PYTHON --version

echo ""
echo "Installing dependencies..."
$PYTHON -m pip install --user torch torchvision torchaudio einops transformers datasets tqdm pandas matplotlib seaborn numpy scikit-learn

echo ""
echo "Installing FlashLinearAttention..."
$PYTHON -m pip install --user flash-linear-attention

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
$PYTHON verify_install.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To run benchmarks with Python 3.11:"
echo "  /usr/bin/python3.11 run_benchmark.py --quick"
echo "  /usr/bin/python3.11 run_benchmark.py --all"
