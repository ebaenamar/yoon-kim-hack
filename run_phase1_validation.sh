#!/bin/bash

# Phase 1 Validation: Extended Training (300 epochs)
# Run experiments sequentially to avoid GPU resource conflicts

echo "=========================================="
echo "PHASE 1 VALIDATION: Extended Training"
echo "=========================================="
echo ""

# Activate conda environment
source ~/anaconda3/bin/activate
conda activate arch_bench

# MQAR Task - All Architectures
echo "=== MQAR Task ==="
echo "Training LSTM..."
python run_benchmark.py --architectures lstm --task mqar --num-epochs 300 > logs/lstm_mqar_300.log 2>&1
echo "✓ LSTM complete"

echo "Training GRU..."
python run_benchmark.py --architectures gru --task mqar --num-epochs 300 > logs/gru_mqar_300.log 2>&1
echo "✓ GRU complete"

echo "Training RoPE..."
python run_benchmark.py --architectures rope --task mqar --num-epochs 300 > logs/rope_mqar_300.log 2>&1
echo "✓ RoPE complete"

echo "Training Vanilla Transformer..."
python run_benchmark.py --architectures vanilla_transformer --task mqar --num-epochs 300 > logs/vanilla_transformer_mqar_300.log 2>&1
echo "✓ Vanilla Transformer complete"

echo ""
echo "=== Flip-Flop Task ==="
echo "Training DeltaNet..."
python run_benchmark.py --architectures deltanet --task flip_flop --num-epochs 300 > logs/deltanet_flipflop_300.log 2>&1
echo "✓ DeltaNet complete"

echo "Training LSTM..."
python run_benchmark.py --architectures lstm --task flip_flop --num-epochs 300 > logs/lstm_flipflop_300.log 2>&1
echo "✓ LSTM complete"

echo "Training RoPE..."
python run_benchmark.py --architectures rope --task flip_flop --num-epochs 300 > logs/rope_flipflop_300.log 2>&1
echo "✓ RoPE complete"

echo ""
echo "=== Knapsack Task ==="
echo "Training LSTM..."
python run_benchmark.py --architectures lstm --task knapsack --num-epochs 300 > logs/lstm_knapsack_300.log 2>&1
echo "✓ LSTM complete"

echo "Training DeltaNet..."
python run_benchmark.py --architectures deltanet --task knapsack --num-epochs 300 > logs/deltanet_knapsack_300.log 2>&1
echo "✓ DeltaNet complete"

echo ""
echo "=========================================="
echo "✅ PHASE 1 VALIDATION COMPLETE!"
echo "=========================================="
echo "Results saved in results/"
