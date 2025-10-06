#!/usr/bin/env python3
"""
Generate plots for fairness analysis and Phase 1 results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import json

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def extract_history_from_log(filepath):
    """Extract training history from log file"""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract accuracy values
    eval_lines = re.findall(r'Eval Loss: ([\d.]+) \| Accuracy: ([\d.]+)', content)
    
    if not eval_lines:
        return None
    
    accuracies = [float(acc) for _, acc in eval_lines]
    losses = [float(loss) for loss, _ in eval_lines]
    
    return {
        'accuracy': accuracies,
        'loss': losses,
        'epochs': list(range(1, len(accuracies) + 1))
    }

# 1. LEARNING CURVES - MQAR
print("Generating Plot 1: Learning Curves (MQAR)...")

fig, ax = plt.subplots(figsize=(14, 8))

models = {
    'DeltaNet': 'deltanet_mqar_300_epochs.log',
    'LSTM': 'logs/lstm_mqar_300.log',
    'GRU': 'logs/gru_mqar_300.log',
    'RoPE': 'logs/rope_mqar_300.log',
    'Vanilla Transformer': 'logs/vanilla_transformer_mqar_300.log'
}

colors = {
    'DeltaNet': '#2ecc71',  # Green
    'LSTM': '#e74c3c',      # Red
    'GRU': '#3498db',       # Blue
    'RoPE': '#f39c12',      # Orange
    'Vanilla Transformer': '#9b59b6'  # Purple
}

for model_name, filepath in models.items():
    history = extract_history_from_log(filepath)
    if history:
        epochs = history['epochs']
        accuracy = [acc * 100 for acc in history['accuracy']]  # Convert to percentage
        
        ax.plot(epochs, accuracy, label=model_name, 
                linewidth=2.5, color=colors.get(model_name, 'gray'),
                alpha=0.8)

# Add convergence line
ax.axhline(y=80, color='red', linestyle='--', linewidth=1.5, 
           label='Convergence Threshold (80%)', alpha=0.5)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Learning Curves: MQAR Task (300 Epochs)\nDeltaNet converges at epoch 18, others never converge', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 300)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('results/plots/learning_curves_mqar.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/learning_curves_mqar.png")

# 2. CONVERGENCE COMPARISON
print("\nGenerating Plot 2: Convergence Comparison...")

fig, ax = plt.subplots(figsize=(12, 8))

convergence_data = {
    'DeltaNet': 18,
    'LSTM': 300,  # Never converged
    'GRU': 300,
    'RoPE': 300,
    'Vanilla Transformer': 300
}

final_accuracy = {
    'DeltaNet': 91.31,
    'LSTM': 0.88,
    'GRU': 2.88,
    'RoPE': 2.19,
    'Vanilla Transformer': 2.97
}

models_list = list(convergence_data.keys())
epochs_to_converge = [convergence_data[m] for m in models_list]
final_acc = [final_accuracy[m] for m in models_list]

# Create bars
bars = ax.barh(models_list, epochs_to_converge, 
               color=['#2ecc71' if e < 300 else '#e74c3c' for e in epochs_to_converge],
               alpha=0.7, edgecolor='black', linewidth=1.5)

# Add accuracy labels
for i, (model, acc) in enumerate(zip(models_list, final_acc)):
    if convergence_data[model] < 300:
        label = f'✓ {acc:.1f}%'
        color = 'green'
    else:
        label = f'✗ {acc:.1f}%'
        color = 'red'
    
    ax.text(epochs_to_converge[i] + 5, i, label, 
            va='center', fontsize=11, fontweight='bold', color=color)

ax.set_xlabel('Epochs to Convergence (>80% accuracy)', fontsize=14, fontweight='bold')
ax.set_title('Convergence Speed Comparison: MQAR Task\nGreen = Converged | Red = Failed to Converge', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 320)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/plots/convergence_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/convergence_comparison.png")

# 3. ACCURACY vs PARAMETERS
print("\nGenerating Plot 3: Accuracy vs Parameters...")

fig, ax = plt.subplots(figsize=(12, 8))

params = {
    'DeltaNet': 1.63,
    'LSTM': 2.14,
    'GRU': 1.64,
    'RoPE': 3.16,
    'Vanilla Transformer': 3.29
}

x = [params[m] for m in models_list]
y = final_acc

scatter = ax.scatter(x, y, s=300, alpha=0.7, 
                     c=[colors.get(m, 'gray') for m in models_list],
                     edgecolors='black', linewidth=2)

# Add labels
for i, model in enumerate(models_list):
    ax.annotate(model, (x[i], y[i]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

ax.set_xlabel('Model Parameters (Millions)', fontsize=14, fontweight='bold')
ax.set_ylabel('Final Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Accuracy vs Model Size: MQAR Task\nMore parameters ≠ Better performance', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 100)

plt.tight_layout()
plt.savefig('results/plots/accuracy_vs_parameters.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/accuracy_vs_parameters.png")

# 4. MULTI-TASK HEATMAP
print("\nGenerating Plot 4: Multi-Task Performance Heatmap...")

# Data from Phase 1 results
task_results = {
    'DeltaNet': {'MQAR': 91.31, 'Flip-Flop': 10.00, 'Knapsack': 0.0},
    'LSTM': {'MQAR': 0.88, 'Flip-Flop': 10.28, 'Knapsack': 1.0},
    'GRU': {'MQAR': 2.88, 'Flip-Flop': 0.0, 'Knapsack': 0.0},
    'RoPE': {'MQAR': 2.19, 'Flip-Flop': 10.68, 'Knapsack': 0.0},
    'Vanilla Transformer': {'MQAR': 2.97, 'Flip-Flop': 0.0, 'Knapsack': 0.0}
}

# Create matrix
tasks = ['MQAR', 'Flip-Flop', 'Knapsack']
models_order = ['DeltaNet', 'LSTM', 'GRU', 'RoPE', 'Vanilla Transformer']

matrix = []
for model in models_order:
    row = [task_results[model].get(task, 0) for task in tasks]
    matrix.append(row)

matrix = np.array(matrix)

fig, ax = plt.subplots(figsize=(10, 8))

# Normalize by column (task) to show relative performance
matrix_normalized = matrix.copy()
for j in range(matrix.shape[1]):
    col_max = matrix[:, j].max()
    if col_max > 0:
        matrix_normalized[:, j] = matrix[:, j] / col_max

sns.heatmap(matrix_normalized, annot=matrix, fmt='.2f', 
            xticklabels=tasks, yticklabels=models_order,
            cmap='RdYlGn', center=0.5, vmin=0, vmax=1,
            cbar_kws={'label': 'Relative Performance'},
            linewidths=2, linecolor='black',
            ax=ax, annot_kws={'fontsize': 12, 'fontweight': 'bold'})

ax.set_title('Multi-Task Performance Heatmap\nValues = Absolute Accuracy (%) | Color = Relative Performance', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Task', fontsize=14, fontweight='bold')
ax.set_ylabel('Architecture', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/plots/multitask_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/multitask_heatmap.png")

# 5. PERPLEXITY CURVES
print("\nGenerating Plot 5: Perplexity Curves (MQAR)...")

fig, ax = plt.subplots(figsize=(14, 8))

for model_name, filepath in models.items():
    history = extract_history_from_log(filepath)
    if history:
        epochs = history['epochs']
        perplexity = [np.exp(loss) for loss in history['loss']]
        
        ax.plot(epochs, perplexity, label=model_name, 
                linewidth=2.5, color=colors.get(model_name, 'gray'),
                alpha=0.8)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Perplexity (lower is better)', fontsize=14, fontweight='bold')
ax.set_title('Perplexity Curves: MQAR Task\nDeltaNet achieves dramatically lower perplexity', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 300)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('results/plots/perplexity_curves_mqar.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/perplexity_curves_mqar.png")

print("\n" + "="*80)
print("✅ All plots generated successfully!")
print("="*80)
print("\nGenerated plots:")
print("  1. learning_curves_mqar.png - Shows convergence behavior")
print("  2. convergence_comparison.png - Epochs to convergence")
print("  3. accuracy_vs_parameters.png - Model size vs performance")
print("  4. multitask_heatmap.png - Performance across all tasks")
print("  5. perplexity_curves_mqar.png - Loss curves")
