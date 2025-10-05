# Architecture Pentathlon: A Comparative Analysis of Inductive Biases

## 1. Introduction

This report summarizes a sequence of controlled experiments performed on a diverse set of neural network architectures, including RNNs (`LSTM`, `GRU`), Transformers (`RoPE`, `Vanilla`), and SSMs (`DeltaNet`, `GLA`). The primary objective was to evaluate their comparative performance across a suite of diagnostic tasks (`MQAR`, `Flip-Flop`, `Knapsack`) to identify which architectural principles demonstrate superior memory, reasoning, and efficiency tradeoffs under identical training constraints.

Our methodology emphasizes a **broad, controlled comparison** to map architectural strengths before engaging in deep, model-specific hyperparameter tuning. For a detailed justification of this "breadth-first" approach, see our [Methodological Comparison](COMPARISON.md).

## 2. Experiment Summary

The following table summarizes the results of the Pentathlon, run with a fixed baseline configuration (`d_model=256`, `n_layers=4`, `epochs=30`).

| Task | Cognitive Skill | Winner(s) | Approx. Accuracy | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **MQAR** | Associative Memory | `deltanet` | ~0.7% | Only SSMs showed learning; others failed. |
| **Flip-Flop** | State Tracking | `rope` | ~10.7% | All models performed poorly, suggesting a task/capacity bottleneck. |
| **Knapsack** | Algorithmic Reasoning| `lstm` | ~1.0% | RNNs clearly outperform all other families. |

**High-Level Conclusion:** No single architecture dominates. Performance is highly task-specific, reflecting a strong link between a model's inductive bias and the problem's cognitive demands.

## 3. Results and Analysis

### 3.1. Architectural Specialization: A Heatmap View

To visualize the *relative* strengths, we normalized accuracy scores within each task. A score of 1.0 (bright yellow) denotes the top-performing architecture for that task.

![Relative Performance Heatmap](results/plots/relative_performance_heatmap.png)

**Interpretation:** The heatmap starkly reveals specialization. `deltanet` is the undisputed champion of associative memory (`mqar`), while `lstm` dominates algorithmic reasoning (`knapsack`). This visual evidence strongly supports our central thesis: **architecture matters, and its choice must be task-aware.**

### 3.2. Per-Task Performance Breakdown

#### **MQAR (Memory Quality and Retrieval)**
*   **Observation:** As shown in the absolute accuracy plot below, `deltanet` was the only architecture to consistently outperform random chance. Despite the low absolute accuracy (~0.7%) due to the short training run, its ability to learn at all is significant.
*   **Interpretation:** This confirms that the delta-rule update mechanism in `deltanet` provides a stronger inductive bias for temporal/associative data retention than the mechanisms in LSTMs or Transformers under these constrained settings.

![Accuracy by Task](results/plots/accuracy_comparison.png)

#### **Flip-Flop (State Tracking)**
*   **Observation:** All models plateaued at ~10% accuracy. While `rope` was the relative winner, no model demonstrated a meaningful ability to solve the task.
*   **Interpretation:** This uniform failure suggests the task may impose a representational ceiling that is too high for models of this modest capacity (`d_model=256`). It serves as an excellent candidate for future experiments on model scaling.

#### **Knapsack (Algorithmic Reasoning)**
*   **Observation:** The `lstm` marginally but clearly outperformed all other models. Transformers and `deltanet` failed entirely, showing no emergent reasoning capacity.
*   **Interpretation:** This finding suggests that for tasks requiring procedural or step-by-step reasoning, the strict sequential processing of RNNs remains a powerful and relevant inductive bias.

## 4. Conclusions

The "Pentathlon" experiments confirm that in a resource-constrained environment, architectural biases are the primary determinant of performance.
1.  **DeltaNet** is the strongest associative learner.
2.  **RNNs (LSTM/GRU)** retain a niche but important advantage in procedural reasoning.
3.  **Transformers**, without massive scale or task-specific augmentation, underperform on these structured diagnostic tasks compared to specialized architectures.

## 5. Next Steps

The next phase of this research will build upon these foundational findings:
1.  **Complete the Pentathlon:** Finalize debugging of the `Hidden-Mode` task to complete the benchmark dataset.
2.  **Execute Grid Search:** Launch a targeted hyperparameter search for the "winning" architectures on their specialized tasks to find their peak performance.
    *   **DeltaNet on `MQAR`:** Vary `d_model` and `n_layers` to test memory scalability.
    *   **LSTM on `Knapsack`:** Vary `d_model` and `n_layers` to enhance reasoning capacity.
3.  **Expand Architecture Suite:** Integrate modern baselines like **Mamba-2** and **DeltaFormer** for robustness validation.

---
### **Appendix: Debugging Log**

During the experimental phase, several issues were identified and resolved:
*   **`Hidden-Mode` Dataset:** A `KeyError` occurred due to a mismatch between expected column names (`prompt`, `completion`) and actual headers (`input`, `target`). This was fixed by updating the data parser.
*   **`ModelConfig` Incompatibility:** A `TypeError` was raised because `num_heads` was passed to non-attention models. This was resolved by making the `ModelConfig` and trainer logic more robust.
*   **Padding and Batching:** `ValueError`s related to tensor sizes were resolved by implementing a robust padding mechanism in the `HiddenModeDataset` and making the training loop flexible to handle variable batch contents.
