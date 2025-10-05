import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
import torch
import numpy as np

from models.model_factory import create_model
from models.base_model import ModelConfig
from tasks import FlipFlopDataset, FlipFlopConfig
from trainer import train_model, SequenceDataset

def tune_weight_decay():
    """Train a model with different weight decay values and save results."""
    # --- Configuration ---
    architecture = 'rope'
    task = 'flip_flop'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 50
    num_runs = 1 # For tuning, we only need one run per setting

    weight_decays_to_test = [0.1, 0.01, 0.0]
    learning_rate = 1e-3 # Fix learning rate
    results = []

    print(f"Tuning Hyperparameters for: Arch={architecture}, Task={task}")
    print(f"Fixed Learning Rate: {learning_rate}")
    print(f"Testing Weight Decays: {weight_decays_to_test}")

    for wd in weight_decays_to_test:
        print(f"\n{'*'*80}")
        print(f"Testing Weight Decay: {wd}")
        print(f"{'*'*80}")

        # --- Setup model and data for each run ---
        task_config = FlipFlopConfig(num_symbols=10, seq_length=256, seed=42)
        model_config = ModelConfig(
            vocab_size=task_config.num_symbols + 2,
            d_model=256,
            n_layers=4,
            max_seq_len=512
        )
        model = create_model(architecture, model_config)
        
        dataset_train = FlipFlopDataset(task_config)
        eval_config = FlipFlopConfig(seed=43)
        dataset_eval = FlipFlopDataset(eval_config)

        train_ds = SequenceDataset(dataset_train, num_samples=1000)
        eval_ds = SequenceDataset(dataset_eval, num_samples=200)

        # --- Train ---
        try:
            history = train_model(
                model,
                train_ds,
                eval_ds,
                num_epochs=num_epochs,
                batch_size=16,
                learning_rate=learning_rate,
                weight_decay=wd, # Use the current weight decay
                device=device
            )

            # --- Save results ---
            final_accuracy = history['eval_accuracy'][-1]
            results.append({'weight_decay': wd, 'final_accuracy': final_accuracy})

            # Save detailed history for this run
            log_dir = Path('results/logs')
            log_dir.mkdir(parents=True, exist_ok=True)
            history_path = log_dir / f'tuning_{architecture}_{task}_wd_{wd}.json'
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"Saved history to {history_path}")

        except Exception as e:
            print(f"Error during training with wd={wd}: {e}")
            results.append({'weight_decay': wd, 'final_accuracy': 0.0})

    # --- Print Final Summary ---
    print(f"\n{'='*80}")
    print("Hyperparameter Tuning Summary")
    print(f"{'='*80}")
    print("Architecture: rope | Task: flip_flop")
    print("----------------------------------------")
    print("Weight Decay    | Final Accuracy")
    print("----------------------------------------")
    for res in sorted(results, key=lambda x: x['final_accuracy'], reverse=True):
        print(f"{res['weight_decay']:<15} | {res['final_accuracy']:.2%}")
    print("----------------------------------------")

if __name__ == '__main__':
    tune_weight_decay()
