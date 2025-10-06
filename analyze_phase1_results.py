#!/usr/bin/env python3
"""
Analyze Phase 1 Validation Results
"""

import pandas as pd
import json

# Read results
df = pd.read_csv('results/benchmark_results.csv')

print("="*80)
print(" PHASE 1 VALIDATION RESULTS - 300 EPOCHS ".center(80))
print("="*80)
print()

# Group by task
for task in df['task'].unique():
    task_df = df[df['task'] == task].copy()
    task_df = task_df.sort_values('accuracy_mean', ascending=False)
    
    print(f"\n{'='*80}")
    print(f" {task.upper()} TASK ".center(80))
    print(f"{'='*80}")
    print()
    
    for _, row in task_df.iterrows():
        arch = row['architecture']
        acc = row['accuracy_mean']
        ppl = row['perplexity_mean']
        time = row['train_time_mean']
        params = row['num_params']
        
        print(f"  {arch:30s} | Acc: {acc:7.4f} | PPL: {ppl:8.2f} | Time: {time:6.1f}s | Params: {params:,}")
    
    print()

print("="*80)
print()

# Summary statistics
print("\n" + "="*80)
print(" KEY FINDINGS ".center(80))
print("="*80)
print()

# Best per task
for task in df['task'].unique():
    task_df = df[df['task'] == task]
    best = task_df.loc[task_df['accuracy_mean'].idxmax()]
    print(f"  {task.upper():20s} | Winner: {best['architecture']:30s} | Accuracy: {best['accuracy_mean']:.4f}")

print()
print("="*80)
