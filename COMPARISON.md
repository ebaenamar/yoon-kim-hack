# Methodological Comparison: Broad vs. Deep Benchmarking

This document contrasts our "Architecture Pentathlon" methodology with an alternative approach focused on deep hyperparameter tuning for a single model class. Both strategies are valid but answer different research questions.

## Our Approach: The "Architecture Pentathlon"

Our methodology is designed as a **broad, comparative survey** to map the cognitive specializations of different architectural *families*.

*   **Objective:** To identify the inherent strengths and weaknesses (inductive biases) of RNNs, Transformers, and SSMs. We ask: "Which *type* of architecture is best suited for this *type* of problem?"
*   **Strategy: Controlled Experiment.** To isolate the impact of architecture, we hold all other variables constant (learning rate, model dimensions, optimizer). This ensures that observed performance differences are attributable to the core architectural design, not to fine-tuning.
*   **Analogy:** This is like testing a sedan, a truck, and a sports car on three different terrains (a racetrack, a mountain road, and a city street) using the *same stock engine and tires* for all of them. The goal is to see which *type* of vehicle is fundamentally better suited to each terrain, not to find the fastest possible version of the sports car.
*   **Outcome:** Our results, visualized in the heatmap, clearly show specialization profiles: SSMs excel at associative recall, while RNNs retain an edge in algorithmic reasoning. This provides a high-level map of the solution space.

## Alternative Approach: Deep Hyperparameter Search

The results shared from the other team exemplify a **deep, granular search** focused on optimizing a single architecture.

*   **Objective:** To find the optimal hyperparameter configuration for a specific model (e.g., `LSTM`) on a specific task. They ask: "What is the absolute best performance we can squeeze out of an `LSTM` on this problem?"
*   **Strategy: Iterative Optimization.** This approach involves a manual or automated search through the hyperparameter space (learning rate, L1 regularization, model width/depth, optimizer betas, etc.). It is not a controlled comparison between architectures but an optimization process for one.
*   **Analogy:** This is like taking the sports car from the previous analogy and spending weeks in the garage tuning its engine, testing different fuel mixtures, and swapping out various tire compounds to achieve the fastest possible lap time on the racetrack.
*   **Outcome:** This process can yield a highly optimized, state-of-the-art result for a specific model-task pair, but it does not provide a broad comparative understanding of *why* other architectures may have failed or succeeded.

## Synthesis and Justification

Our "Pentathlon" approach is the logical and necessary **first step** in a comprehensive research program. It efficiently maps the landscape and identifies the most promising architecture-task pairings. The deep hyperparameter search is the crucial **second step**, where we take the "winners" from our broad survey and optimize them for peak performance.

By starting broad, we avoid prematurely optimizing an architecture that is fundamentally ill-suited for a given problem (e.g., spending weeks tuning a Transformer for the `Knapsack` task, where our results show it is unlikely to succeed). Our methodology provides the foundational map that makes a subsequent deep dive both efficient and scientifically justified.
