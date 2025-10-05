"""
Model wrappers using FlashLinearAttention library
Supports: Mamba2, DeltaNet, Gated DeltaNet, RWKV-7, etc.
"""

import torch
import torch.nn as nn
from .base_model import BaseLanguageModel, ModelConfig

try:
    # Import FLA layers
    from fla.layers import (
        DeltaNet,
        GatedDeltaNet, 
        RWKV7Attention,
        MultiScaleRetention,
        GatedLinearAttention,
    )
    from fla.models import Mamba2Model, Mamba2Config
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    print("Warning: FlashLinearAttention not installed. Install with: pip install flash-linear-attention")


class FLALanguageModel(BaseLanguageModel):
    """Language model using FLA layers"""
    
    def __init__(self, config: ModelConfig, layer_type: str = "deltanet"):
        super().__init__(config)
        
        if not FLA_AVAILABLE:
            raise ImportError("FlashLinearAttention library not available")
        
        self.layer_type = layer_type
        
        # Build layers based on architecture
        for _ in range(config.n_layers):
            layer = self._create_layer(config)
            self.layers.append(layer)
    
    def _create_layer(self, config: ModelConfig) -> nn.Module:
        """Create a layer based on architecture type"""
        
        if self.layer_type == "deltanet":
            return DeltaNetLayer(config.d_model, config.dropout)
        elif self.layer_type == "gated_deltanet":
            return GatedDeltaNetLayer(config.d_model, config.dropout)
        elif self.layer_type == "rwkv7":
            return RWKV7Layer(config.d_model, config.dropout)
        elif self.layer_type == "retention":
            return RetentionLayer(config.d_model, config.dropout)
        elif self.layer_type == "gla":
            return GLALayer(config.d_model, config.dropout)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")


class DeltaNetLayer(nn.Module):
    """DeltaNet layer wrapper"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.deltanet = DeltaNet(
            d_model=d_model,
            expand_k=1.0,
            expand_v=2.0,
            num_heads=4,
            mode='chunk'
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual
        normed = self.norm(x)
        out = self.deltanet(normed)[0]  # DeltaNet returns (output, None)
        return x + self.dropout(out)


class GatedDeltaNetLayer(nn.Module):
    """Gated DeltaNet layer wrapper"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gated_deltanet = GatedDeltaNet(
            hidden_size=d_model,
            expand_k=1.0,
            expand_v=2.0,
            num_heads=4,
            mode='chunk'
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        out = self.gated_deltanet(normed)[0]
        return x + self.dropout(out)


class RWKV7Layer(nn.Module):
    """RWKV-7 layer wrapper"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.rwkv7 = RWKV7Attention(
            d_model=d_model,
            expand_k=1.0,
            expand_v=2.0,
            num_heads=4,
            mode='chunk'
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        out = self.rwkv7(normed)[0]
        return x + self.dropout(out)


class RetentionLayer(nn.Module):
    """Multi-Scale Retention layer wrapper"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.retention = MultiScaleRetention(
            d_model=d_model,
            expand_k=1.0,
            expand_v=2.0,
            num_heads=4,
            mode='chunk'
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        out = self.retention(normed)[0]
        return x + self.dropout(out)


class GLALayer(nn.Module):
    """Gated Linear Attention layer wrapper"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gla = GatedLinearAttention(
            hidden_size=d_model,
            expand_k=1.0,
            expand_v=2.0,
            num_heads=4,
            mode='chunk'
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        out = self.gla(normed)[0]
        return x + self.dropout(out)


def create_fla_model(config: ModelConfig, architecture: str) -> BaseLanguageModel:
    """
    Factory function to create FLA models
    
    Args:
        config: Model configuration
        architecture: One of: deltanet, gated_deltanet, rwkv7, retention, gla
        
    Returns:
        FLA language model
    """
    return FLALanguageModel(config, layer_type=architecture)
