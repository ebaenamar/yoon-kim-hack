# Architecture Comparison - Technical Notes

## Architecture Categories

### 1. Classic RNNs (Baseline)
**LSTM & GRU**
- **Strengths**: Simple, well-understood, stable training
- **Weaknesses**: Limited long-range dependencies, sequential processing
- **Expected Performance**: 
  - Good on simple tasks
  - Struggle with MQAR (limited memory capacity)
  - Moderate on Flip-Flop

### 2. Transformers (Strong Baseline)
**RoPE Transformer**
- **Strengths**: Strong compositional reasoning, proven performance
- **Weaknesses**: Quadratic complexity, requires positional encoding
- **Expected Performance**:
  - Excellent on MQAR (attention mechanism ideal)
  - Good on Flip-Flop
  - Best on Knapsack (compositional DP reasoning)

**Vanilla Transformer**
- **Strengths**: Standard architecture, interpretable
- **Weaknesses**: Fixed positional embeddings, length generalization issues
- **Expected Performance**: Similar to RoPE but slightly worse on long sequences

### 3. Modern Linear RNNs (Research Focus)

**DeltaNet**
- **Key Innovation**: Delta rule for linear transformers
- **Complexity**: Linear in sequence length
- **Expected Performance**: 
  - Competitive with Transformers on MQAR
  - Strong on associative tasks

**Gated DeltaNet**  
- **Key Innovation**: Gating mechanism + delta rule
- **Relation**: Improvement over Mamba2
- **Expected Performance**: 
  - Best in class for linear RNNs
  - Should match or exceed DeltaNet

**RWKV-7**
- **Key Innovation**: Receptance-weighted key-value with matrix states
- **Strengths**: Parallelizable, strong empirical results
- **Expected Performance**:
  - Excellent on Flip-Flop (designed for state tracking)
  - Competitive on MQAR

**Retention (RetNet)**
- **Key Innovation**: Retention mechanism as alternative to attention
- **Strengths**: Training parallelizable, inference efficient
- **Expected Performance**:
  - Strong on MQAR (retention = simplified attention)
  - Good all-around performance

**GLA (Gated Linear Attention)**
- **Key Innovation**: Gating + linear attention
- **Strengths**: Hardware efficient, strong empirical results
- **Expected Performance**:
  - Competitive with Transformers
  - Strong on associative tasks

## Task-Architecture Affinity Matrix

### MQAR (Associative Recall)
**Difficulty**: Medium  
**Key Requirement**: Memory for key-value pairs

Expected Ranking:
1. RoPE / Retention (attention-based)
2. DeltaNet / Gated DeltaNet (delta rule)
3. GLA / RWKV-7 (linear attention)
4. LSTM / GRU (limited capacity)

### Flip-Flop (State Tracking)
**Difficulty**: Medium-Hard  
**Key Requirement**: Long-range state maintenance

Expected Ranking:
1. RWKV-7 (designed for this)
2. RoPE / Transformers (global context)
3. Retention / GLA (memory mechanisms)
4. DeltaNet / Gated DeltaNet (sequential)
5. LSTM / GRU (limited range)

### Knapsack (DP Reasoning)
**Difficulty**: Hard  
**Key Requirement**: Compositional reasoning, max operations

Expected Ranking:
1. RoPE / Transformers (compositional)
2. All others (will likely struggle)

**Note**: This is the hardest task. Don't expect high accuracy!

## Computational Complexity

| Architecture | Training | Inference | Memory |
|-------------|----------|-----------|--------|
| LSTM/GRU    | O(n)     | O(n)      | O(1)   |
| Transformers| O(n²)    | O(n²)     | O(n²)  |
| DeltaNet    | O(n)     | O(n)      | O(1)   |
| Gated ΔNet  | O(n)     | O(n)      | O(1)   |
| RWKV-7      | O(n)     | O(1)      | O(1)   |
| Retention   | O(n²)→O(n)| O(1)     | O(1)   |
| GLA         | O(n)     | O(n)      | O(1)   |

## Model Sizes (Approximate)

For d_model=256, n_layers=4, vocab_size=128:

- **LSTM**: ~270K parameters
- **GRU**: ~270K parameters  
- **RoPE Transformer**: ~400K parameters
- **Vanilla Transformer**: ~400K parameters
- **DeltaNet**: ~300K parameters
- **Gated DeltaNet**: ~320K parameters
- **RWKV-7**: ~310K parameters
- **Retention**: ~350K parameters
- **GLA**: ~330K parameters

All models are small enough to train quickly on A100.

## Training Characteristics

### Fast Converging
- Transformers (strong inductive bias)
- Retention (effective learning)

### Moderate Converging
- GLA, DeltaNet (need more epochs)
- RWKV-7 (complex dynamics)

### Slow Converging
- LSTM/GRU (limited capacity)
- May need more data/epochs

## Known Challenges

### MQAR Task
- **Challenge**: Vocabulary size vs num_kv_pairs tradeoff
- **Solution**: Ensure vocab_size > num_kv_pairs

### Flip-Flop Task
- **Challenge**: State collapse (forgetting old states)
- **Solution**: Longer training, careful hyperparameters

### Knapsack Task  
- **Challenge**: Requires exact max operations
- **Solution**: This is a hard task - expect low accuracy across all architectures

## Hyperparameter Notes

Current settings (in run_benchmark.py):
- Learning rate: 3e-4 (good for all)
- Batch size: 16 (can reduce if OOM)
- Epochs: 10-15 (may need more for harder tasks)
- Gradient clip: 1.0 (stable training)

### If accuracy is low:
1. Increase epochs (20-30)
2. Reduce learning rate (1e-4)
3. Increase model size (d_model=512)
4. Generate more training data

### If training is slow:
1. Reduce batch size
2. Reduce model size
3. Use FLA optimized kernels

## Research Questions

### Primary Questions:
1. **Do modern linear RNNs close the gap with Transformers?**
   - Compare on MQAR accuracy

2. **Is there a tradeoff between efficiency and performance?**
   - Plot accuracy vs training time

3. **Which inductive biases work best for which tasks?**
   - Architecture-task affinity analysis

### Secondary Questions:
4. **How well do architectures generalize to longer sequences?**
   - Test with increased seq_length

5. **Can any architecture learn true algorithmic reasoning?**
   - Knapsack task is the key test

6. **What is the scaling behavior?**
   - Vary model size and see performance

## Expected Timeline

- **Quick test**: 2-3 minutes
- **Single architecture-task**: 5-15 minutes
- **Baseline (3 archs, 3 tasks)**: 30-45 minutes
- **Full benchmark (9 archs, 3 tasks)**: 2-3 hours
- **Extended analysis**: +1-2 hours

Total: **~4-6 hours** for complete study

## Success Criteria

**Minimum Viable Results:**
- At least one architecture achieves >80% on MQAR
- Clear performance differences between architectures
- Efficiency-performance tradeoffs visible

**Strong Results:**
- Multiple architectures >90% on MQAR
- >50% on Flip-Flop
- >10% on Knapsack (this would be impressive!)

**Publication-Worthy Results:**
- Complete comparison across all architectures
- Statistical significance testing
- Novel insights about architecture capabilities
- Clear recommendations for practitioners
