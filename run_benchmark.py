"""
Main benchmark runner
Orchestrates training and evaluation across all architectures and tasks
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from models.model_factory import create_model, list_available_architectures
from models.base_model import ModelConfig
from tasks import (
    MQARDataset, MQARConfig,
    FlipFlopDataset, FlipFlopConfig,
    TropicalTaskConfig, KnapsackTask,
    HiddenModeConfig, HiddenModeDataset
)
from trainer import train_model, SequenceDataset

def setup_directories():
    """Create results directories"""
    dirs = ['results', 'results/plots', 'results/checkpoints', 'results/logs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def benchmark_mqar(architecture: str, device: str, seed: int, num_epochs: int, learning_rate: float, weight_decay: float, num_kv_pairs: int, d_model: int, n_layers: int, num_heads: int) -> Dict:
    """Benchmark on MQAR task"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {architecture} on MQAR")
    print(f"{'='*60}")
    
    # New rules for the stress test
    max_seq_len = 4 * num_kv_pairs
    vocab_size = max_seq_len + 1

    print(f"Stress Test Config: num_kv_pairs={num_kv_pairs}, max_seq_len={max_seq_len}, vocab_size={vocab_size}")

    task_config = MQARConfig(vocab_size=vocab_size, num_kv_pairs=num_kv_pairs, num_queries=16, seed=seed)
    model_config = ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )
    
    model = create_model(architecture, model_config)
    if architecture == 'deltanet':
        print("Converting deltanet model to bfloat16")
        model = model.to(dtype=torch.bfloat16)
        
    dataset_train = MQARDataset(task_config)
    eval_config = MQARConfig(
        vocab_size=vocab_size,
        num_kv_pairs=num_kv_pairs,
        num_queries=16,
        seed=seed + 1
    )
    dataset_eval = MQARDataset(eval_config)
    
    print(f"Model: {architecture} | Parameters: {model.get_num_params():,}")
    
    train_ds = SequenceDataset(dataset_train, num_samples=1000)
    eval_ds = SequenceDataset(dataset_eval, num_samples=200)

    start_time = time.time()
    history = train_model(
        model, 
        train_ds, 
        eval_ds, 
        num_epochs=num_epochs,
        batch_size=16,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    train_time = time.time() - start_time
    
    final_acc = history['eval_accuracy'][-1]
    final_ppl = history['eval_perplexity'][-1]
    
    return {
        'architecture': architecture, 'task': 'mqar', 'accuracy': final_acc,
        'perplexity': final_ppl, 'train_time': train_time,
        'num_params': model.get_num_params(), 'history': history
    }

def benchmark_flip_flop(architecture: str, device: str, seed: int, num_epochs: int, learning_rate: float, weight_decay: float, d_model: int, n_layers: int, num_heads: int) -> Dict:
    """Benchmark on Flip-Flop task"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {architecture} on Flip-Flop")
    print(f"{'='*60}")
    
    task_config = FlipFlopConfig(num_symbols=10, seq_length=256, seed=seed)
    model_config = ModelConfig(
        vocab_size=task_config.num_symbols + 2,
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads,
        max_seq_len=512
    )
    
    model = create_model(architecture, model_config)
    if architecture == 'deltanet':
        print("Converting deltanet model to bfloat16")
        model = model.to(dtype=torch.bfloat16)

    dataset_train = FlipFlopDataset(task_config)
    eval_config = FlipFlopConfig(
        num_symbols=task_config.num_symbols,
        seq_length=task_config.seq_length,
        seed=seed + 1
    )
    dataset_eval = FlipFlopDataset(eval_config)
    
    print(f"Model: {architecture} | Parameters: {model.get_num_params():,}")
    
    train_ds = SequenceDataset(dataset_train, num_samples=1000)
    eval_ds = SequenceDataset(dataset_eval, num_samples=200)

    start_time = time.time()
    history = train_model(
        model, 
        train_ds, 
        eval_ds, 
        num_epochs=num_epochs,
        batch_size=16,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    train_time = time.time() - start_time
    
    final_acc = history['eval_accuracy'][-1]
    final_ppl = history['eval_perplexity'][-1]
    
    return {
        'architecture': architecture, 'task': 'flip_flop', 'accuracy': final_acc,
        'perplexity': final_ppl, 'train_time': train_time,
        'num_params': model.get_num_params(), 'history': history
    }

def benchmark_knapsack(architecture: str, device: str, seed: int, num_epochs: int, learning_rate: float, weight_decay: float, d_model: int, n_layers: int, num_heads: int) -> Dict:
    """Benchmark on Knapsack task"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {architecture} on Knapsack")
    print(f"{'='*60}")
    
    task_config = TropicalTaskConfig(seed=seed)
    model_config = ModelConfig(
        vocab_size=4096,
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads,
        max_seq_len=512
    )
    
    model = create_model(architecture, model_config)
    if architecture == 'deltanet':
        print("Converting deltanet model to bfloat16")
        model = model.to(dtype=torch.bfloat16)

    task_train = KnapsackTask(task_config)
    eval_config = TropicalTaskConfig(seed=seed + 1)
    task_eval = KnapsackTask(eval_config)
    
    print(f"Model: {architecture} | Parameters: {model.get_num_params():,}")
    
    train_ds = SequenceDataset(task_train, num_samples=1000)
    eval_ds = SequenceDataset(task_eval, num_samples=200)
    
    start_time = time.time()
    history = train_model(
        model, 
        train_ds, 
        eval_ds, 
        use_padding=True,
        num_epochs=num_epochs,
        batch_size=16,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    train_time = time.time() - start_time
    
    final_acc = history['eval_accuracy'][-1]
    final_ppl = history['eval_perplexity'][-1]
    
    return {
        'architecture': architecture, 'task': 'knapsack', 'accuracy': final_acc,
        'perplexity': final_ppl, 'train_time': train_time,
        'num_params': model.get_num_params(), 'history': history
    }

def benchmark_hidden_mode(architecture: str, device: str, seed: int, num_epochs: int, learning_rate: float, weight_decay: float, d_model: int, n_layers: int, num_heads: int) -> Dict:
    """Benchmark on Hidden-Mode task"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {architecture} on Hidden-Mode")
    print(f"{'='*60}")

    task_config = HiddenModeConfig(
        url='https://media.githubusercontent.com/media/jopetty/word-problem/refs/heads/main/data/A5%3D10.csv',
        seed=seed
    )
    
    model_config = ModelConfig(
        vocab_size=64, # Placeholder, will be updated by dataset
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads,
        max_seq_len=64 # Set a fixed max_seq_len for this task
    )

    # Create a single dataset and then split it
    full_dataset = HiddenModeDataset(task_config, max_seq_len=model_config.max_seq_len)
    train_size = int(0.8 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    dataset_train, dataset_eval = torch.utils.data.random_split(full_dataset, [train_size, eval_size])
    
    # Now update model_config with the actual vocab_size from the dataset
    model_config.vocab_size = full_dataset.vocab_size

    model = create_model(architecture, model_config)
    
    # Note: SequenceDataset is not needed here as HiddenModeDataset already returns tensors
    train_ds = dataset_train
    eval_ds = dataset_eval

    start_time = time.time()
    history = train_model(
        model, 
        train_ds, 
        eval_ds, 
        num_epochs=num_epochs,
        batch_size=16, # Let the trainer handle the DataLoader
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    train_time = time.time() - start_time
    
    final_acc = history['eval_accuracy'][-1]
    final_ppl = history['eval_perplexity'][-1]
    
    return {
        'architecture': architecture, 'task': 'hidden_mode', 'accuracy': final_acc,
        'perplexity': final_ppl, 'train_time': train_time,
        'num_params': model.get_num_params(), 'history': history
    }

def run_experiment(results: List[Dict], architecture: str, task: str, device: str, num_runs: int, num_epochs: int, learning_rate: float, weight_decay: float, num_kv_pairs: int, d_model: int, n_layers: int, num_heads: int):
    """Runs a single experiment (arch-task pair) multiple times for statistical significance."""
    run_metrics = {'accuracy': [], 'perplexity': [], 'train_time': []}
    num_params = 0
    all_histories = []

    print(f"\n{'*'*80}")
    print(f"Running Experiment: Arch={architecture}, Task={task}, Runs={num_runs}, Epochs={num_epochs}")
    print(f"{'*'*80}")

    for i in range(num_runs):
        current_seed = 42 + i
        print(f"\n--- Run {i+1}/{num_runs} (Seed: {current_seed}) ---")
        
        try:
            if task == 'mqar':
                run_results = benchmark_mqar(architecture, device, seed=current_seed, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, num_kv_pairs=num_kv_pairs, d_model=d_model, n_layers=n_layers, num_heads=num_heads)
            elif task == 'flip_flop':
                run_results = benchmark_flip_flop(architecture, device, seed=current_seed, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, d_model=d_model, n_layers=n_layers, num_heads=num_heads)
            elif task == 'knapsack':
                run_results = benchmark_knapsack(architecture, device, seed=current_seed, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, d_model=d_model, n_layers=n_layers, num_heads=num_heads)
            elif task == 'hidden_mode':
                run_results = benchmark_hidden_mode(architecture, device, seed=current_seed, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, d_model=d_model, n_layers=n_layers, num_heads=num_heads)
            else:
                print(f"Unknown task: {task}")
                return

            run_metrics['accuracy'].append(run_results['accuracy'])
            run_metrics['perplexity'].append(run_results['perplexity'])
            run_metrics['train_time'].append(run_results['train_time'])
            all_histories.append(run_results['history'])
            num_params = run_results['num_params']

        except Exception as e:
            print(f"\n❌ Error during run {i+1} for {architecture} on {task}: {e}")

    if not run_metrics['accuracy']:
        print(f"\n❌ All runs failed for {architecture} on {task}. Skipping.")
        return

    # For grid search, create a unique name for the architecture based on its params
    arch_name_for_results = f"{architecture}_d{d_model}_n{n_layers}_h{num_heads}"
    task_name_for_results = f"{task}_kv{num_kv_pairs}" if task == 'mqar' else task

    results.append({
        'architecture': arch_name_for_results, 'task': task_name_for_results,
        'accuracy_mean': np.mean(run_metrics['accuracy']), 'accuracy_std': np.std(run_metrics['accuracy']),
        'perplexity_mean': np.mean(run_metrics['perplexity']), 'perplexity_std': np.std(run_metrics['perplexity']),
        'train_time_mean': np.mean(run_metrics['train_time']), 'train_time_std': np.std(run_metrics['train_time']),
        'num_params': num_params, 'num_runs': num_runs, 'histories': all_histories
    })

def save_results(results: List[Dict]):
    """Save aggregated results to CSV and plots"""
    df = pd.DataFrame(results)
    csv_path = "results/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    for result in results:
        arch = result['architecture']
        task = result['task']
        history_path = f"results/logs/{arch}_{task}_history.json"
        with open(history_path, 'w') as f:
            json.dump(result['histories'], f, indent=2)

def print_summary(results: List[Dict]):
    """Prints a formatted summary of the benchmark results."""
    df = pd.DataFrame(results)
    print("\n" + "="*79)
    print("="*30 + " BENCHMARK RESULTS SUMMARY " + "="*28)
    print("="*79)

    for task in df['task'].unique():
        print(f"\n{task.upper()} Task (runs: {df['num_runs'].iloc[0]}):")
        task_df = df[df['task'] == task].sort_values(by='accuracy_mean', ascending=False)
        
        print_df = pd.DataFrame()
        print_df['architecture'] = task_df['architecture']
        print_df['accuracy'] = task_df.apply(lambda r: f"{r['accuracy_mean']:.4f} (±{r['accuracy_std']:.4f})", axis=1)
        print_df['perplexity'] = task_df.apply(lambda r: f"{r['perplexity_mean']:.2f} (±{r['perplexity_std']:.2f})", axis=1)
        print_df['train_time'] = task_df.apply(lambda r: f"{r['train_time_mean']:.2f}s (±{r['train_time_std']:.2f}s)", axis=1)
        print_df['num_params'] = task_df['num_params']

        print(print_df.to_string(index=False))
    
    print("\n" + "="*79)

def plot_results(results: List[Dict]):
    """Generate and save plots for the results."""
    df = pd.DataFrame(results)
    sns.set_theme(style="whitegrid")

    # New plot for MQAR Stress Test
    mqar_df = df[df['task'] == 'mqar'].copy()
    # The 'architecture' column in the results CSV will have suffixes like '_kv64', we need to clean it
    # and create a proper num_kv_pairs column.
    if not mqar_df.empty:
        mqar_df['num_kv_pairs'] = mqar_df['architecture'].apply(lambda x: int(x.split('_kv')[-1]) if '_kv' in x else 0)
        mqar_df['architecture'] = mqar_df['architecture'].apply(lambda x: x.split('_kv')[0] if '_kv' in x else x)
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=mqar_df, x='num_kv_pairs', y='accuracy_mean', hue='architecture', marker='o')
        plt.title('MQAR Stress Test: Accuracy vs. Number of Key-Value Pairs', fontsize=16)
        plt.xlabel('Number of Key-Value Pairs (Difficulty)')
        plt.ylabel('Mean Accuracy')
        plt.xscale('log')
        plt.xticks(mqar_df['num_kv_pairs'].unique(), mqar_df['num_kv_pairs'].unique())
        plt.grid(True, which="both", ls="--")
        plt.legend(title='Architecture')
        plt.savefig("results/plots/mqar_stress_test.png")
        print("Saved: results/plots/mqar_stress_test.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle('Accuracy by Architecture and Task', fontsize=16)
    
    tasks = df['task'].unique()
    for i, task in enumerate(tasks):
        task_df = df[df['task'] == task].sort_values(by='accuracy_mean', ascending=True)
        axes[i].barh(
            task_df['architecture'], task_df['accuracy_mean'], 
            xerr=task_df['accuracy_std'], capsize=4
        )
        axes[i].set_title(f"{task.upper()} Task")
        axes[i].set_xlabel("Accuracy")
        axes[i].set_xlim(0, max(1.0, df['accuracy_mean'].max() * 1.1 if not df.empty else 1.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/plots/accuracy_comparison.png")
    print("Saved: results/plots/accuracy_comparison.png")

    plt.figure(figsize=(12, 8))
    for task in tasks:
        task_df = df[df['task'] == task]
        plt.errorbar(
            x=task_df['train_time_mean'], y=task_df['accuracy_mean'],
            xerr=task_df['train_time_std'], yerr=task_df['accuracy_std'],
            fmt='o', capsize=5, label=task
        )
    plt.title("Efficiency: Accuracy vs Training Time")
    plt.xlabel("Mean Training Time (seconds)")
    plt.ylabel("Mean Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/efficiency_comparison.png")
    print("Saved: results/plots/efficiency_comparison.png")

    plt.figure(figsize=(15, 7))
    sns.barplot(data=df, x='architecture', y='perplexity_mean', hue='task')
    plt.title("Perplexity by Architecture and Task")
    plt.ylabel("Perplexity (lower is better)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/plots/perplexity_comparison.png")
    print("Saved: results/plots/perplexity_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='Architecture Benchmark on Diagnostic Tasks')
    parser.add_argument('--architectures', type=str, default='all', help='Comma-separated list of architectures or "all"')
    parser.add_argument('--tasks', type=str, default='all', help='Comma-separated list of tasks: mqar,flip_flop,knapsack,hidden_mode or "all"')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--quick', action='store_true', help='Quick run with fewer architectures for testing')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs for each experiment for statistical significance')
    parser.add_argument('--num-epochs', type=int, default=15, help='Number of training epochs for each run')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Optimizer weight decay')
    parser.add_argument('--num-kv-pairs', type=str, default='32', help='Comma-separated list of key-value pairs for MQAR stress test')

    # Grid search arguments
    parser.add_argument('--d-models', type=str, default='256', help='Comma-separated list of d_model values for grid search')
    parser.add_argument('--n-layers-list', type=str, default='4', help='Comma-separated list of n_layers values for grid search')
    parser.add_argument('--num-heads-list', type=str, default='4', help='Comma-separated list of num_heads values for grid search')
    
    args = parser.parse_args()
    
    setup_directories()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    available_archs = list_available_architectures()
    print(f"\nAvailable architectures: {', '.join(available_archs)}")

    if args.quick:
        architectures = ['lstm', 'gru', 'rope']
        print(f"\nQuick mode: testing {architectures}")
    elif args.architectures == 'all':
        architectures = sorted(available_archs)
    else:
        architectures = args.architectures.split(',')

    all_tasks = ['mqar', 'flip_flop', 'knapsack', 'hidden_mode']
    if args.tasks == 'all':
        tasks = all_tasks
    else:
        tasks = args.tasks.split(',')

    print(f"\nRunning benchmark:")
    print(f"  Architectures: {architectures}")
    print(f"  Tasks: {tasks}")

    results = []
    d_models = [int(x) for x in args.d_models.split(',')]
    n_layers_list = [int(x) for x in args.n_layers_list.split(',')]
    num_heads_list = [int(x) for x in args.num_heads_list.split(',')]

    # Main experiment loop for grid search
    for arch in architectures:
        for task in tasks:
            for d_model in d_models:
                for n_layers in n_layers_list:
                    for num_heads in num_heads_list:
                        if task == 'mqar':
                            num_kv_pairs_list = [int(x) for x in str(args.num_kv_pairs).split(',')]
                            for num_kv in num_kv_pairs_list:
                                run_experiment(results, arch, task, device, args.num_runs, args.num_epochs, args.learning_rate, args.weight_decay, num_kv, d_model, n_layers, num_heads)
                        else:
                            # For non-mqar tasks, num_kv_pairs is not used, pass a default
                            run_experiment(results, arch, task, device, args.num_runs, args.num_epochs, args.learning_rate, args.weight_decay, 0, d_model, n_layers, num_heads)

    if results:
        save_results(results)
        print_summary(results)
        plot_results(results)
        print(f"\n✅ Benchmark complete! Results saved in results/")
    else:
        print("\n❌ No results generated. Check for errors above.")

if __name__ == '__main__':
    main()