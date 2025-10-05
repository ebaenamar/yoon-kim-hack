"""
Classic RNN models: LSTM and GRU
"""

import torch
import torch.nn as nn
from .base_model import BaseLanguageModel, ModelConfig


class LSTMLanguageModel(BaseLanguageModel):
    """LSTM-based language model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0,
            batch_first=True
        )
        
        # Override layers (we use LSTM directly)
        self.layers = nn.ModuleList([self.lstm])
        
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
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Project to vocab
        logits = self.output_proj(x)
        
        return logits


class GRULanguageModel(BaseLanguageModel):
    """GRU-based language model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0,
            batch_first=True
        )
        
        # Override layers
        self.layers = nn.ModuleList([self.gru])
        
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass"""
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # GRU
        x, _ = self.gru(x)
        
        # Project to vocab
        logits = self.output_proj(x)
        
        return logits


def create_lstm_model(config: ModelConfig) -> LSTMLanguageModel:
    """Create LSTM model"""
    return LSTMLanguageModel(config)


def create_gru_model(config: ModelConfig) -> GRULanguageModel:
    """Create GRU model"""
    return GRULanguageModel(config)
