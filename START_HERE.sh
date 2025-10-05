#!/bin/bash

echo "=========================================="
echo "Architecture Benchmark - Quick Start"
echo "=========================================="
echo ""

cd /home/nuwins-server1/CascadeProjects/architecture-benchmark

echo "Current directory: $(pwd)"
echo ""

echo "Step 1: Testing setup..."
python quick_start.py

echo ""
echo "=========================================="
echo "Setup verified! Choose your experiment:"
echo "=========================================="
echo ""
echo "Option 1 - Quick Baseline (30 min):"
echo "  python run_benchmark.py --architectures lstm,gru,rope --tasks mqar,flip_flop,knapsack"
echo ""
echo "Option 2 - Install FLA for modern architectures:"
echo "  bash install.sh"
echo ""
echo "Option 3 - Full benchmark (requires FLA, 2-3 hours):"
echo "  python run_benchmark.py --all"
echo ""
echo "Results will be saved to: results/"
echo ""
