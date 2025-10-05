import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_relative_performance_heatmap(df):
    """Plots a heatmap of relative performance to highlight specialization."""
    # Clean up task names for better readability
    df['task'] = df['task'].str.replace('_kv64', '', regex=False)

    # Create a pivot table: architectures vs. tasks
    pivot_df = df.pivot_table(index='architecture', columns='task', values='accuracy_mean')

    # Normalize the performance for each task (column-wise)
    # This shows how each model performs relative to the best model for that specific task
    normalized_df = pivot_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else x, axis=0)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_df, annot=True, cmap="viridis", fmt=".2f")
    plt.title('Normalized Performance Heatmap (Relative Strength by Task)', fontsize=16)
    plt.xlabel('Diagnostic Task')
    plt.ylabel('Architecture')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_path = "results/plots/relative_performance_heatmap.png"
    plt.savefig(plot_path)
    print(f"Saved: {plot_path}")
    plt.close()

def generate_plots():
    """Reads the benchmark results and generates insightful plots."""
    try:
        df = pd.read_csv('results/pentathlon_results_partial.csv')
    except FileNotFoundError:
        print("Error: results/pentathlon_results_partial.csv not found.")
        return

    # Generate the new heatmap
    plot_relative_performance_heatmap(df.copy())

    sns.set_theme(style="whitegrid")

    # --- Plot 1: MQAR STRESS Test (The 'Breaking Point' Curve) ---
    mqar_df = df[df['task'].str.startswith('mqar')].copy()
    if not mqar_df.empty:
        # Reconstruct the num_kv_pairs column from the raw data.
        # We know the experiments were run in order [64, 256, 512] for each architecture.
        kv_levels = [64, 256, 512]
        reconstructed_dfs = []
        for arch in mqar_df['architecture'].unique():
            arch_df = mqar_df[mqar_df['architecture'] == arch].copy()
            if len(arch_df) == len(kv_levels):
                arch_df['num_kv_pairs'] = kv_levels
                reconstructed_dfs.append(arch_df)
            else:
                print(f"[Warning] Found {len(arch_df)} MQAR runs for {arch}, expected {len(kv_levels)}. Skipping this arch for the stress test plot.")
        
        if not reconstructed_dfs:
            print("[Info] Not enough data for MQAR stress test plot. Skipping.")
        else:
            mqar_df = pd.concat(reconstructed_dfs)
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=mqar_df, x='num_kv_pairs', y='accuracy_mean', hue='architecture', marker='o', sort=False)
            plt.title('MQAR Stress Test: Accuracy vs. Task Difficulty', fontsize=16)
            plt.xlabel('Number of Key-Value Pairs (Difficulty)')
            plt.ylabel('Mean Accuracy')
            plt.xscale('log')
            plt.xticks(mqar_df['num_kv_pairs'].unique(), labels=mqar_df['num_kv_pairs'].unique().astype(str))
            plt.minorticks_off()
            plt.grid(True, which="both", ls="--")
            plt.legend(title='Architecture')
            
            plot_path = "results/plots/mqar_stress_test_curve.png"
            plt.savefig(plot_path)
            print(f"Saved: {plot_path}")
            plt.close()

    # --- Plot 2: Accuracy by Architecture and Task (The one you showed me) ---
    # For the general accuracy plot, we need to identify the 64 kv-pair run for each architecture
    mqar_df_for_plot = df[df['task'] == 'mqar'].copy()
    mqar_64_indices = []
    for arch in mqar_df_for_plot['architecture'].unique():
        arch_indices = mqar_df_for_plot[mqar_df_for_plot['architecture'] == arch].index
        if len(arch_indices) > 0:
            mqar_64_indices.append(arch_indices[0]) # The first entry for each arch is the 64 kv run
    
    # Filter out the harder mqar tasks, keeping only the 64 kv run and all other tasks
    plot_df = df[(df['task'] != 'mqar') | (df.index.isin(mqar_64_indices))].copy()
    tasks = plot_df['task'].unique()
    
    if len(tasks) > 0:
        num_tasks = len(tasks)
        fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 5), sharey=False)
        if num_tasks == 1:
            axes = [axes] # Make it iterable if there's only one plot
        fig.suptitle('Accuracy by Architecture and Task', fontsize=16)
        
        for i, task in enumerate(tasks):
            task_df = plot_df[plot_df['task'] == task].sort_values(by='accuracy_mean', ascending=True)
            axes[i].barh(
                task_df['architecture'], task_df['accuracy_mean'], 
                xerr=task_df['accuracy_std'], capsize=4
            )
            axes[i].set_title(f"{task.upper()} Task")
            axes[i].set_xlabel("Accuracy")
            axes[i].set_xlim(0, 1.0)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = "results/plots/accuracy_by_task_desaggregated.png"
        plt.savefig(plot_path)
        print(f"Saved: {plot_path}")
        plt.close()

    # --- Existing plotting functions can remain as they are ---
    # You can call them here if you still want the old plots, for example:
    # plot_accuracy_comparison(df)
    # plot_efficiency_comparison(df)


if __name__ == '__main__':
    generate_plots()
