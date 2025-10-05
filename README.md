# Architecture Performance Benchmark on Diagnostic Algorithmic Tasks

## Overview
This project implements a comprehensive benchmark comparing modern architectures (SSMs, Linear RNNs, Transformer variants) on challenging synthetic diagnostic tasks.

## Research Question
How do modern architectures (vs. vanilla Transformers) perform on harder synthetic tasks?

## Architectures Tested

### Modern Linear RNNs
- Mamba-2
- DeltaNet
- Gated DeltaNet
- RWKV-7
- MesaNet

### Classic RNNs
- LSTM
- GRU

### Transformer Variants
- RoPE (baseline)
- PaTH Attention
- DeltaFormer
- Native Sparse Attention
- 2-Simplicial Attention

## Benchmark Tasks

### 1. MQAR (Multi-Query Associative Recall)
Tests the ability to recall key-value mappings across long sequences.

### 2. Flip-Flop Language Modeling
Evaluates long-range dependency tracking by requiring models to copy binary symbols while ignoring intervening tokens.

### 3. Tropical Attention Tasks
Dynamic programming-style combinatorial optimization problems that require sharp, scale-invariant reasoning.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run full benchmark
```bash
python run_benchmark.py --all
```

### Run specific task
```bash
python run_benchmark.py --task mqar --architectures mamba2,deltanet,rope
```

### Run with custom parameters
```bash
python run_benchmark.py --task mqar --seq_length 2048 --vocab_size 128 --num_samples 1000
```

## Metrics

- **Accuracy**: Exact match accuracy on task outputs
- **Perplexity**: Language modeling quality
- **Length Generalization**: Performance on longer sequences than training
- **Value Generalization**: Performance on larger vocabularies than training
- **Throughput**: Tokens processed per second
- **Memory Usage**: Peak GPU memory consumption

## Results

Results are saved in `results/` directory with:
- CSV files with detailed metrics per architecture
- Comparison plots and visualizations
- Statistical significance tests

## References

- Tropical Attention Paper: https://arxiv.org/abs/2505.17190
- FlashLinearAttention: https://github.com/fla-org/flash-linear-attention
- MQAR Analysis: https://hazyresearch.stanford.edu/blog/2024-06-22-ac
- Flip-Flop: https://arxiv.org/abs/2306.00946
