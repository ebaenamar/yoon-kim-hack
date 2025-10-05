#!/bin/bash

echo "=========================================="
echo "Installing FlashLinearAttention via Conda"
echo "=========================================="

# Source conda
source ~/anaconda3/bin/activate

# Create environment with Python 3.10
echo "Creating conda environment 'arch_bench' with Python 3.10..."
conda create -n arch_bench python=3.10 -y

# Activate
conda activate arch_bench

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA 12.1..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
echo "Installing other dependencies..."
pip install einops transformers datasets tqdm pandas matplotlib seaborn scikit-learn numpy

# Install FlashLinearAttention
echo "Installing FlashLinearAttention..."
pip install flash-linear-attention

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "To use the full benchmark with all 9 architectures:"
echo ""
echo "  conda activate arch_bench"
echo "  cd /home/nuwins-server1/CascadeProjects/architecture-benchmark"
echo "  python run_benchmark.py --all"
echo ""
echo "To verify:"
echo "  conda activate arch_bench"
echo "  python verify_install.py"
echo ""
