# METHODOLOGY.md (v2.0)

## 1. Research Objective & Core Questions

The primary objective of this research is to move beyond monolithic benchmarks and conduct a granular, comparative analysis of different neural network architectures on a suite of diagnostic tasks. Each task is designed to isolate and test a specific cognitive capability, allowing us to map the strengths and weaknesses of each architectural family.

Our core research questions are:
1.  **Generalist vs. Specialist:** Is there a single "master" architecture that excels across all tasks, or is high performance intrinsically linked to specialization?
2.  **Architectural Trade-offs:** What are the fundamental trade-offs between architectural families (e.g., training parallelism vs. inference efficiency, memory capacity vs. algorithmic reasoning)?
3.  **Cognitive Profiling:** Can we create a "cognitive profile" for each architecture, identifying the types of problems it is naturally suited to solve?

## 2. Architectures Under Test

We selected representative models from three dominant families of sequence modeling to ensure a broad and meaningful comparison.

*   **Recurrent Neural Networks (RNNs):** These models process sequences token-by-token, maintaining a compressed, evolving hidden state. They are inherently sequential.
    *   `lstm`: Long Short-Term Memory. The industry standard for RNNs, featuring complex gating mechanisms to combat vanishing gradients.
    *   `gru`: Gated Recurrent Unit. A simplified variant of the LSTM with fewer parameters.

*   **Attention-Based Models (Transformers):** These models use a self-attention mechanism to compute interactions between all pairs of tokens in a sequence simultaneously. They are inherently parallelizable but have a computational cost that scales quadratically with sequence length.
    *   `vanilla_transformer`: A standard Transformer implementation.
    *   `rope`: A Transformer augmented with Rotary Position Embeddings, a more advanced technique for encoding token positions.

*   **Linear Recurrence / State-Space Models (SSMs):** This modern family of models seeks to combine the strengths of both RNNs and Transformers. They can be formulated as a parallel operation for efficient, parallelizable training (like Transformers) but can also be expressed as a recurrent operation for highly efficient, constant-memory inference (like RNNs).
    *   `deltanet`, `gated_deltanet`, `rwkv7`: Models based on the "Delta Rule" principle, a form of linear recurrence.
    *   `gla`: Gated Linear Attention, another variant in this family.

## 3. Diagnostic Task Suite

We curated a "Pentatlón de Arquitecturas" composed of tasks designed to probe distinct capabilities.

### Task 1: Multi-Query Associative Recall (`mqar_kv64`)
*   **Cognitive Capability:** Associative Memory. This task directly measures a model's ability to store a set of arbitrary key-value pairs presented in-context and accurately recall the value for a given key. It is a fundamental test of factual recall and robustness to interference.
*   **Data & Success Metric:** The data is synthetically generated. A sequence contains `N` unique key-value pairs. The model is then prompted with a key and must predict the corresponding value. Accuracy is the primary metric. For this experiment, `N=64`.

### Task 2: Flip-Flop (`flip_flop`)
*   **Cognitive Capability:** State Tracking and Long-Range Memory. The model must track the state of multiple "flip-flops". Each time a symbol `A` appears, the state of flip-flop `A` is toggled. The model's final task is to report the final state of a specific flip-flop. This requires remembering information over long distances and updating internal states correctly.
*   **Data & Success Metric:** Data is synthetically generated. Accuracy is the metric.

### Task 3: Knapsack Problem (`knapsack`)
*   **Cognitive Capability:** Algorithmic Reasoning and Optimization. This task presents the model with a variant of the classic Knapsack problem, a well-known NP-hard optimization problem. Success requires the model to learn an implicit dynamic programming-like algorithm to select items that maximize value without exceeding a weight constraint.
*   **Data & Success Metric:** Data is synthetically generated. Accuracy measures whether the model found the optimal solution.

## 4. Experimental Design & Configuration

### The "Pentathlon" Experiment
*   **Objective:** To create a broad performance baseline and identify areas of specialization for each architecture.
*   **Methodology:** All architectures were run on the three tasks described above (`mqar_kv64`, `flip_flop`, `knapsack`).
*   **Baseline Hyperparameters:** To ensure a fair comparison and isolate architectural differences, all models were initialized with a consistent, modest set of hyperparameters. This allows us to attribute performance differences to fundamental architectural principles rather than extensive, model-specific tuning.
    *   `d_model`: 256
    *   `n_layers`: 4
    *   `num_heads`: 4 (for attention-based models)
*   **Training Configuration:**
    *   `num_epochs`: 30
    *   `optimizer`: AdamW
    *   `learning_rate`: 1e-3
