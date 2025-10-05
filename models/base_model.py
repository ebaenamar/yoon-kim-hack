"""
Base model interface for all architectures
Provides unified training and evaluation interface
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass


class ModelConfig:
    """Configuration for language models"""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, max_seq_len: int, num_heads: int = 4, dropout: float = 0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class BaseLanguageModel(nn.Module):
    """Base class for all language models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Architecture-specific layers (to be defined in subclasses)
        self.layers = nn.ModuleList()
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.output_proj.weight = self.token_embedding.weight
        
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        # Project to vocab
        logits = self.output_proj(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_architecture_name(self) -> str:
        """Get architecture name"""
        return self.config.architecture


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }
