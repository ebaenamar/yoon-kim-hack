import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_convergence(history_file: str, output_image: str):
    """Loads training history and plots the evaluation accuracy curve."""
    try:
        with open(history_file, 'r') as f:
            # The history is a list of dicts, one per run. We take the first.
            history_data = json.load(f)
            if not history_data:
                print(f"Error: History file '{history_file}' is empty.")
                return
            # Assuming single run, so we take the first element of the list.
            run_history = history_data[0]

        eval_accuracy = run_history.get('eval_accuracy')

        if not eval_accuracy:
            print(f"Error: 'eval_accuracy' not found in {history_file}")
            return

        epochs = range(1, len(eval_accuracy) + 1)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot the accuracy curve
        ax.plot(epochs, eval_accuracy, marker='o', linestyle='-', label='Eval Accuracy')

        # Highlight the "grokking" moment
        # Find the point of maximum gradient (fastest learning)
        grad = np.gradient(eval_accuracy)
        grokking_epoch = np.argmax(grad) + 1
        grokking_acc = eval_accuracy[grokking_epoch - 1]
        ax.axvline(x=grokking_epoch, color='r', linestyle='--', label=f'"Grokking" Point (Epoch {grokking_epoch})')
        ax.annotate(f'Epoch {grokking_epoch}\nAccuracy: {grokking_acc:.2%}', 
                    xy=(grokking_epoch, grokking_acc), 
                    xytext=(grokking_epoch + 5, grokking_acc - 0.1), 
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8))

        # Find the sweet spot (e.g., where accuracy reaches 95% of its max)
        max_acc = max(eval_accuracy)
        sweet_spot_threshold = 0.95 * max_acc
        try:
            sweet_spot_epoch = next(i for i, acc in enumerate(eval_accuracy) if acc >= sweet_spot_threshold) + 1
            ax.axvline(x=sweet_spot_epoch, color='g', linestyle='--', label=f'"Sweet Spot" (Epoch {sweet_spot_epoch})')
        except StopIteration:
            sweet_spot_epoch = len(epochs) # Default to last epoch if threshold not met

        ax.set_title('Convergence Analysis: rope on flip_flop Task', fontsize=16)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Evaluation Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_image)
        print(f"Convergence analysis plot saved to {output_image}")

    except FileNotFoundError:
        print(f"Error: History file not found at {history_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    analyze_convergence(
        history_file='results/logs/rope_flip_flop_history.json',
        output_image='results/plots/convergence_analysis.png'
    )
