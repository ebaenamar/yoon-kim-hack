# Running Experiments - Step by Step Guide

## ✅ System Verified and Working

Quick test shows:
- GPU: NVIDIA A100 80GB (detected)
- All 9 architectures available
- Training pipeline functional
- Model parameters: ~260K-400K per architecture

## 🚀 Recommended Experiment Sequence

### Phase 1: Baseline Establishment (30 minutes)
Test classic architectures that work without FLA:

```bash
cd /home/nuwins-server1/CascadeProjects/architecture-benchmark

# Run baseline benchmark
python run_benchmark.py --architectures lstm,gru,rope --tasks mqar,flip_flop,knapsack
```

**Expected Output:**
- 9 model-task combinations (3 architectures × 3 tasks)
- Results in `results/benchmark_results.csv`
- Plots in `results/plots/`

### Phase 2: Install Modern Architectures (5 minutes)

```bash
# Install FlashLinearAttention
pip install -U git+https://github.com/fla-org/flash-linear-attention

# Verify installation
python -c "from fla.layers import DeltaNet; print('FLA installed successfully!')"
```

### Phase 3: Modern Linear RNN Benchmark (1-2 hours)

```bash
# Test modern architectures
python run_benchmark.py --architectures deltanet,gated_deltanet,rwkv7 --tasks mqar,flip_flop,knapsack
```

### Phase 4: Complete Comparison (2-3 hours)

```bash
# Run all architectures on all tasks
python run_benchmark.py --all
```

This will generate:
- **27 experiments** (9 architectures × 3 tasks)
- Complete performance comparison
- Statistical significance analysis
- Publication-ready plots

## 📊 Analyzing Results

### 1. Quick Review
```bash
# View CSV results
cat results/benchmark_results.csv | column -t -s,

# Or open in spreadsheet
libreoffice results/benchmark_results.csv
```

### 2. Key Metrics to Compare

**Accuracy**: Higher is better
- Which architecture solves which task best?
- Are there architecture-task affinities?

**Perplexity**: Lower is better  
- Language modeling quality
- Generalization capability

**Training Time**: Lower is better
- Efficiency consideration
- Practical deployment tradeoffs

**Parameters**: Lower is better (efficiency)
- Memory footprint
- Computational cost

### 3. Research Questions to Explore

1. **Do modern linear RNNs match Transformers on associative recall (MQAR)?**
   - Compare LSTM vs DeltaNet vs RoPE on MQAR accuracy

2. **Which architecture handles long-range dependencies best (Flip-Flop)?**
   - Compare all architectures on Flip-Flop task

3. **Can any architecture learn dynamic programming (Knapsack)?**
   - This is the hardest task - which architectures show non-zero accuracy?

4. **Efficiency-Performance Tradeoffs**
   - Plot accuracy vs training time
   - Find Pareto frontier of architectures

## 🔬 Advanced Experiments

### Experiment 1: Length Generalization
Train on short sequences, test on longer ones:

```python
# Modify run_benchmark.py or create custom script
# Train: seq_len=256, Test: seq_len=512, 1024
```

### Experiment 2: Vocabulary Generalization  
Train on small vocab, test on larger:

```python
# Train: vocab_size=64, Test: vocab_size=128, 256
```

### Experiment 3: Difficulty Scaling
Vary task difficulty:

```python
# MQAR: Increase num_kv_pairs from 32 to 64, 128
# Flip-Flop: Increase num_symbols from 10 to 20, 50
```

## 📈 Expected Findings

Based on literature and architectural properties:

**MQAR (Associative Recall):**
- **Expected Winners**: RoPE, Retention, DeltaNet
- **Expected Struggles**: LSTM, GRU (limited memory)

**Flip-Flop (Long-Range Dependencies):**
- **Expected Winners**: RWKV-7, GLA, RoPE
- **Expected Struggles**: Vanilla Transformer (positional bias)

**Knapsack (DP Reasoning):**
- **Expected Winners**: Transformers (compositional reasoning)
- **Expected Struggles**: All architectures (very hard task)

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size in run_benchmark.py
# Change batch_size=16 to batch_size=8
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Ensure CUDA is being used
python -c "import torch; print(torch.cuda.is_available())"
```

### FLA Installation Issues
```bash
# Try installing with specific versions
pip install torch==2.5.0
pip install triton==3.0.0
pip install -U git+https://github.com/fla-org/flash-linear-attention
```

## 📝 Publishing Results

### 1. Generate Report
```python
# Create analysis notebook
jupyter notebook results_analysis.ipynb
```

### 2. Key Visualizations
- Accuracy comparison bar charts
- Efficiency scatter plots (accuracy vs time)
- Perplexity heatmaps
- Architecture-task affinity matrix

### 3. Statistical Tests
```python
import pandas as pd
from scipy import stats

df = pd.read_csv('results/benchmark_results.csv')

# Compare architectures on MQAR
mqar_results = df[df['task'] == 'mqar']
# Run ANOVA or t-tests
```

## 🎯 Next Steps After Benchmark

1. **Paper Writing**: Document findings
2. **Extended Tasks**: Add more diagnostic tasks
3. **Larger Models**: Scale up to 1B parameters
4. **Real Datasets**: Test on real algorithmic reasoning benchmarks
5. **Open Source**: Share framework with community

## 💡 Tips for Success

1. **Start Small**: Run quick benchmark first
2. **Monitor Progress**: Use `nvidia-smi` to watch GPU usage
3. **Save Checkpoints**: Results auto-saved after each architecture
4. **Compare Incrementally**: Analyze results as they complete
5. **Document**: Take notes on interesting findings

## 🎉 You're Ready!

Everything is set up and verified. Start with Phase 1 (baseline) and work your way through!

```bash
# Start now:
python run_benchmark.py --architectures lstm,gru,rope --tasks mqar,flip_flop
```
