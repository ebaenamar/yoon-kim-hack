"""
Training utilities for language models
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Callable
from tqdm import tqdm
import time


class SequenceDataset(Dataset):
    """Dataset wrapper for sequence tasks"""
    
    def __init__(self, task_generator, num_samples: int):
        self.task_generator = task_generator
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.task_generator.generate_sample()


def train_epoch(model, dataloader, optimizer, device='cuda', grad_clip: float = 1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    for batch in tqdm(dataloader, desc="Training"):
        if len(batch) == 3:
            inputs, targets, masks = batch
        else:
            inputs, targets = batch
            masks = torch.ones_like(inputs, dtype=torch.bool) # Default mask

        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device) # Ensure mask is on the correct device
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(inputs)
        
        # Compute loss only on masked positions
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss = loss.view(targets.shape)
        
        # Apply mask
        masked_loss = (loss * masks).sum() / masks.sum()
        
        # Backward
        masked_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += masked_loss.item() * masks.sum().item()
        total_tokens += masks.sum().item()
    
    return total_loss / total_tokens


def evaluate(model, dataloader, device='cuda'):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 3:
                inputs, targets, masks = batch
            else:
                inputs, targets = batch
                masks = torch.ones_like(inputs, dtype=torch.bool) # Default mask

            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device) # Ensure mask is on the correct device
            
            # Forward
            logits = model(inputs)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = loss.view(targets.shape)
            masked_loss = (loss * masks).sum()
            
            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            correct = ((predictions == targets) & masks).sum()
            
            total_loss += masked_loss.item()
            total_correct += correct.item()
            total_tokens += masks.sum().item()
    
    return {
        'loss': total_loss / total_tokens,
        'accuracy': total_correct / total_tokens,
        'perplexity': torch.exp(torch.tensor(total_loss / total_tokens)).item()
    }


def train_model(
    model,
    train_dataset,
    eval_dataset,
    use_padding: bool = False,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cuda',
    log_interval: int = 1,
    weight_decay: float = 0.01
) -> Dict:
    """
    Train a model on a task
    
    Returns:
        Training history
    """
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    history = {'train_loss': [], 'eval_loss': [], 'eval_accuracy': [], 'eval_perplexity': []}
    
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_time = time.time() - start_time
        
        # Evaluate
        eval_metrics = evaluate(model, eval_loader, device)
        
        # Log
        print(f"Train Loss: {train_loss:.4f} | Time: {train_time:.2f}s")
        print(f"Eval Loss: {eval_metrics['loss']:.4f} | "
              f"Accuracy: {eval_metrics['accuracy']:.4f} | "
              f"Perplexity: {eval_metrics['perplexity']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['eval_loss'].append(eval_metrics['loss'])
        history['eval_accuracy'].append(eval_metrics['accuracy'])
        history['eval_perplexity'].append(eval_metrics['perplexity'])
    
    return history


def collate_fn(batch):
    """Collate function for batching"""
    inputs, targets, masks = zip(*batch)
    
    # Find max length
    max_len = max(inp.size(0) for inp in inputs)
    batch_size = len(inputs)
    
    # Pad
    padded_inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_targets = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, (inp, tgt, mask) in enumerate(zip(inputs, targets, masks)):
        length = inp.size(0)
        padded_inputs[i, :length] = inp
        padded_targets[i, :length] = tgt
        padded_masks[i, :length] = mask
    
    return padded_inputs, padded_targets, padded_masks
