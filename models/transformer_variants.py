"""
Transformer variants: RoPE, PaTH Attention, etc.
"""

import torch
import torch.nn as nn
import math
from .base_model import BaseLanguageModel, ModelConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, seq_len: int):
        """Apply rotary embedding"""
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEAttentionLayer(nn.Module):
    """Transformer layer with RoPE"""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Self-attention with RoPE
        normed = self.norm1(x)
        qkv = self.qkv(normed).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        cos, sin = self.rotary_emb(x, seq_len)
        cos = cos[None, None, :, :].to(q.dtype)
        sin = sin[None, None, :, :].to(q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention (causal)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float('-inf'))
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)
        
        x = x + self.dropout(attn_out)
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class RoPETransformerModel(BaseLanguageModel):
    """Transformer with RoPE positional encoding"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Build layers
        self.layers = nn.ModuleList([
            RoPEAttentionLayer(config.d_model, num_heads=4, dropout=config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # Project to vocab
        logits = self.output_proj(x)
        
        return logits


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer without position encoding"""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Self-attention
        normed = self.norm1(x)
        qkv = self.qkv(normed).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float('-inf'))
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)
        
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        
        return x


class VanillaTransformerModel(BaseLanguageModel):
    """Vanilla Transformer (baseline)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model))
        
        # Build layers
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(config.d_model, num_heads=4, dropout=config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        
        # Embed tokens + add position
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # Project to vocab
        logits = self.output_proj(x)
        
        return logits


def create_rope_model(config: ModelConfig) -> RoPETransformerModel:
    """Create RoPE Transformer model"""
    return RoPETransformerModel(config)


def create_vanilla_transformer(config: ModelConfig) -> VanillaTransformerModel:
    """Create vanilla Transformer model"""
    return VanillaTransformerModel(config)
