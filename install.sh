#!/bin/bash

echo "=========================================="
echo "Architecture Benchmark Setup"
echo "=========================================="

# Check CUDA
echo -e "\nChecking CUDA..."
nvidia-smi

# Install base requirements
echo -e "\nInstalling base requirements..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt

# Install FLA from source (recommended for latest features)
echo -e "\nInstalling FlashLinearAttention..."
pip uninstall fla-core flash-linear-attention -y
pip install -U git+https://github.com/fla-org/flash-linear-attention

echo -e "\n=========================================="
echo "Installation complete!"
echo "=========================================="
echo -e "\nRun quick test:"
echo "  python quick_start.py"
echo -e "\nRun benchmark:"
echo "  python run_benchmark.py --quick"
