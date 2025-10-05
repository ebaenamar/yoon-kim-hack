"""
Flip-Flop Language Modeling (FFLM) Task
Tests long-range dependency tracking by requiring models to copy binary symbols
while ignoring intervening tokens.

Example: A 1 B 0 C 1 D 1 E 0 -> predict the next A, B, C, D, E values
The model needs to track the last value seen for each symbol.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class FlipFlopConfig:
    """Configuration for Flip-Flop task"""
    num_symbols: int = 10  # Number of different symbols (A, B, C, ...)
    seq_length: int = 512
    flip_prob: float = 0.5  # Probability of flipping a symbol's value
    seed: int = 42


class FlipFlopDataset:
    """Generate Flip-Flop dataset"""
    
    def __init__(self, config: FlipFlopConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        # Token IDs: 0-9 are symbols, 10-11 are binary values
        self.symbol_offset = 0
        self.value_offset = config.num_symbols
        
    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a single flip-flop sample
        
        Returns:
            input_ids: [seq_len] - Input sequence with symbol-value pairs
            target_ids: [seq_len] - Target sequence (next token prediction)
            mask: [seq_len] - Mask indicating positions to evaluate
        """
        config = self.config
        
        # Initialize state for each symbol (current value)
        symbol_states = {i: None for i in range(config.num_symbols)}
        
        sequence = []
        targets = []
        
        for _ in range(config.seq_length // 2):
            # Sample a symbol
            symbol = self.rng.randint(0, config.num_symbols)
            
            # Decide whether to flip
            if symbol_states[symbol] is None or self.rng.random() < config.flip_prob:
                # Assign or flip value
                value = self.rng.randint(0, 2)
                symbol_states[symbol] = value
            else:
                # Keep current value
                value = symbol_states[symbol]
            
            # Add symbol and value to sequence
            symbol_token = self.symbol_offset + symbol
            value_token = self.value_offset + value
            
            sequence.extend([symbol_token, value_token])
            
        # Create targets (shifted by 1)
        targets = sequence[1:] + [0]  # Padding at the end
        
        # Mask: evaluate on value tokens (odd positions)
        mask = torch.zeros(len(sequence), dtype=torch.bool)
        mask[1::2] = True  # Value positions
        
        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            mask
        )
    
    def generate_long_range_sample(self, distance: int = 256) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate sample with controlled long-range dependencies
        
        Args:
            distance: Minimum distance between symbol occurrences
        """
        config = self.config
        
        sequence = []
        symbol_last_seen = {}
        
        for _ in range(config.seq_length // 2):
            # Choose symbol ensuring minimum distance
            valid_symbols = [
                s for s in range(config.num_symbols)
                if s not in symbol_last_seen or 
                (len(sequence) - symbol_last_seen[s]) >= distance
            ]
            
            if not valid_symbols:
                valid_symbols = list(range(config.num_symbols))
            
            symbol = self.rng.choice(valid_symbols)
            
            # Determine value
            if symbol not in symbol_last_seen:
                value = self.rng.randint(0, 2)
            else:
                # Flip or keep
                if self.rng.random() < config.flip_prob:
                    value = self.rng.randint(0, 2)
                else:
                    # Keep last value - this creates long-range dependency
                    last_pos = symbol_last_seen[symbol]
                    value = (sequence[last_pos + 1] - self.value_offset)
            
            symbol_token = self.symbol_offset + symbol
            value_token = self.value_offset + value
            
            sequence.extend([symbol_token, value_token])
            symbol_last_seen[symbol] = len(sequence) - 2
        
        targets = sequence[1:] + [0]
        mask = torch.zeros(len(sequence), dtype=torch.bool)
        mask[1::2] = True
        
        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            mask
        )
    
    def generate_batch(self, batch_size: int, long_range: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of flip-flop samples"""
        batch_inputs = []
        batch_targets = []
        batch_masks = []
        
        for _ in range(batch_size):
            if long_range:
                inp, tgt, mask = self.generate_long_range_sample()
            else:
                inp, tgt, mask = self.generate_sample()
            batch_inputs.append(inp)
            batch_targets.append(tgt)
            batch_masks.append(mask)
        
        # Stack
        return (
            torch.stack(batch_inputs),
            torch.stack(batch_targets),
            torch.stack(batch_masks)
        )


def evaluate_flip_flop(model, dataset: FlipFlopDataset, num_samples: int = 100, 
                      long_range: bool = False, device='cuda') -> dict:
    """
    Evaluate model on Flip-Flop task
    
    Args:
        model: Model with forward(input_ids) -> logits method
        dataset: FlipFlopDataset instance
        num_samples: Number of samples to evaluate
        long_range: Whether to test long-range dependencies
        device: Device to run on
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    total_correct = 0
    total_predictions = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            if long_range:
                inputs, targets, mask = dataset.generate_long_range_sample()
            else:
                inputs, targets, mask = dataset.generate_sample()
                
            inputs = inputs.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            # Get model predictions
            logits = model(inputs)
            predictions = logits.argmax(dim=-1)
            
            # Compute accuracy on value positions
            masked_preds = predictions[mask]
            masked_targets = targets[mask]
            
            total_correct += (masked_preds == masked_targets).sum().item()
            total_predictions += mask.sum().item()
    
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_predictions': total_predictions,
        'task': 'flip_flop',
        'long_range': long_range
    }


def test_flip_flop():
    """Test Flip-Flop dataset generation"""
    config = FlipFlopConfig(num_symbols=5, seq_length=20)
    dataset = FlipFlopDataset(config)
    
    inputs, targets, mask = dataset.generate_sample()
    print(f"Input sequence: {inputs}")
    print(f"Target sequence: {targets}")
    print(f"Evaluation mask: {mask}")
    print(f"Sequence length: {len(inputs)}")
    
    # Test long-range
    inputs_lr, targets_lr, mask_lr = dataset.generate_long_range_sample(distance=8)
    print(f"\nLong-range input: {inputs_lr}")
    
    # Generate batch
    batch_inputs, batch_targets, batch_masks = dataset.generate_batch(4)
    print(f"\nBatch shape: {batch_inputs.shape}")


if __name__ == '__main__':
    test_flip_flop()
