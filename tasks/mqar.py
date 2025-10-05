"""
Multi-Query Associative Recall (MQAR) Task
Tests the ability to recall key-value mappings across long sequences.

Example: A 4 B 3 C 6 F 1 E 2 -> A ? C ? F ? E ? 
Expected output: 4 6 1 2
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class MQARConfig:
    """Configuration for MQAR task"""
    vocab_size: int = 128
    num_kv_pairs: int = 64
    num_queries: int = 32
    power_law: float = 0.01  # For query distribution
    seed: int = 42


class MQARDataset:
    """Generate MQAR dataset"""
    
    def __init__(self, config: MQARConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a single MQAR sample
        
        Returns:
            input_ids: [seq_len] - Input sequence with KV pairs and queries
            target_ids: [seq_len] - Target sequence (shifted input with answers)
            mask: [seq_len] - Mask indicating positions to evaluate (query answers)
        """
        config = self.config
        
        # Reserve the last token for the query marker '?'
        content_vocab_size = config.vocab_size - 1
        query_marker = content_vocab_size

        # Sample unique keys and values from the content vocabulary
        keys = self.rng.choice(content_vocab_size, config.num_kv_pairs, replace=False)
        values = self.rng.choice(content_vocab_size, config.num_kv_pairs, replace=False)
        
        # Build KV pairs (interleaved: K1 V1 K2 V2 ...)
        kv_sequence = []
        kv_dict = {}
        for k, v in zip(keys, values):
            kv_sequence.extend([k, v])
            kv_dict[k] = v
        
        # Sample queries with power law distribution
        query_weights = np.arange(1, config.num_kv_pairs + 1) ** (-config.power_law)
        query_weights /= query_weights.sum()
        query_indices = self.rng.choice(
            config.num_kv_pairs, 
            config.num_queries, 
            replace=True, 
            p=query_weights
        )
        
        # Build query sequence (K1 ? K2 ? ...)
        query_sequence = []
        answer_sequence = []
        for idx in query_indices:
            k = keys[idx]
            v = values[idx]
            query_sequence.extend([k, query_marker])
            answer_sequence.extend([k, v])
        
        # Full sequence: KV pairs + query sequence
        full_sequence = kv_sequence + query_sequence
        target_sequence = kv_sequence + answer_sequence
        
        # Mask: only evaluate query answer positions
        mask = torch.zeros(len(full_sequence), dtype=torch.bool)
        query_start = len(kv_sequence)
        for i in range(query_start + 1, len(full_sequence), 2):
            mask[i] = True
        
        return (
            torch.tensor(full_sequence, dtype=torch.long),
            torch.tensor(target_sequence, dtype=torch.long),
            mask
        )
    
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of MQAR samples"""
        batch_inputs = []
        batch_targets = []
        batch_masks = []
        
        for _ in range(batch_size):
            inp, tgt, mask = self.generate_sample()
            batch_inputs.append(inp)
            batch_targets.append(tgt)
            batch_masks.append(mask)
        
        # Pad to same length
        max_len = max(len(inp) for inp in batch_inputs)
        
        padded_inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_targets = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        for i in range(batch_size):
            length = len(batch_inputs[i])
            padded_inputs[i, :length] = batch_inputs[i]
            padded_targets[i, :length] = batch_targets[i]
            padded_masks[i, :length] = batch_masks[i]
        
        return padded_inputs, padded_targets, padded_masks


def evaluate_mqar(model, dataset: MQARDataset, num_samples: int = 100, device='cuda') -> dict:
    """
    Evaluate model on MQAR task
    
    Args:
        model: Model with forward(input_ids) -> logits method
        dataset: MQARDataset instance
        num_samples: Number of samples to evaluate
        device: Device to run on
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    total_correct = 0
    total_queries = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            inputs, targets, mask = dataset.generate_sample()
            inputs = inputs.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            # Get model predictions
            logits = model(inputs)
            predictions = logits.argmax(dim=-1)
            
            # Compute accuracy on query positions
            masked_preds = predictions[mask]
            masked_targets = targets[mask]
            
            total_correct += (masked_preds == masked_targets).sum().item()
            total_queries += mask.sum().item()
    
    accuracy = total_correct / total_queries if total_queries > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_queries': total_queries,
        'task': 'mqar'
    }


def test_mqar():
    """Test MQAR dataset generation"""
    config = MQARConfig(vocab_size=20, num_kv_pairs=5, num_queries=3)
    dataset = MQARDataset(config)
    
    inputs, targets, mask = dataset.generate_sample()
    print(f"Input sequence: {inputs}")
    print(f"Target sequence: {targets}")
    print(f"Evaluation mask: {mask}")
    print(f"Sequence length: {len(inputs)}")
    print(f"Number of queries: {mask.sum().item()}")
    
    # Generate batch
    batch_inputs, batch_targets, batch_masks = dataset.generate_batch(4)
    print(f"\nBatch shape: {batch_inputs.shape}")
    

if __name__ == '__main__':
    test_mqar()
