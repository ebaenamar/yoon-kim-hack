"""
Quick start script to test the benchmark setup
Tests a single architecture on a small MQAR task
"""

import torch
from models.model_factory import create_model, list_available_architectures
from models.base_model import ModelConfig
from tasks import MQARDataset, MQARConfig
from trainer import train_model, SequenceDataset

def quick_test():
    """Quick test of the benchmark system"""
    
    print("="*60)
    print("ARCHITECTURE BENCHMARK - QUICK TEST")
    print("="*60)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # List available architectures
    archs = list_available_architectures()
    print(f"\nAvailable architectures: {', '.join(archs)}")
    
    # Test with LSTM (most compatible)
    architecture = 'lstm'
    print(f"\nTesting with: {architecture}")
    
    # Create small MQAR task
    task_config = MQARConfig(
        vocab_size=32,
        num_kv_pairs=8,
        num_queries=4
    )
    
    model_config = ModelConfig(
        vocab_size=task_config.vocab_size + 1,
        d_model=128,
        n_layers=2,
        max_seq_len=256
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(architecture, model_config)
    print(f"Parameters: {model.get_num_params():,}")
    
    # Create datasets
    print("\nGenerating training data...")
    dataset_train = MQARDataset(task_config)
    eval_config = MQARConfig(
        vocab_size=task_config.vocab_size,
        num_kv_pairs=task_config.num_kv_pairs,
        num_queries=task_config.num_queries,
        seed=999
    )
    dataset_eval = MQARDataset(eval_config)
    
    train_ds = SequenceDataset(dataset_train, num_samples=100)
    eval_ds = SequenceDataset(dataset_eval, num_samples=20)
    
    # Quick training
    print("\nTraining for 3 epochs...")
    history = train_model(
        model,
        train_ds,
        eval_ds,
        num_epochs=3,
        batch_size=8,
        learning_rate=1e-3,
        device=device
    )
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Final Accuracy: {history['eval_accuracy'][-1]:.4f}")
    print(f"Final Perplexity: {history['eval_perplexity'][-1]:.4f}")
    print(f"Final Loss: {history['eval_loss'][-1]:.4f}")
    
    print("\n✅ Quick test completed successfully!")
    print("\nNext steps:")
    print("  1. Install FLA: pip install flash-linear-attention")
    print("  2. Run full benchmark: python run_benchmark.py --quick")
    print("  3. Run all architectures: python run_benchmark.py --all")

if __name__ == '__main__':
    quick_test()
