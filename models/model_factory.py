"""
Factory to create all models with unified interface
"""

from .base_model import ModelConfig, BaseLanguageModel
from .classic_rnn import create_lstm_model, create_gru_model
from .transformer_variants import create_rope_model, create_vanilla_transformer

# Try importing FLA models
try:
    from .fla_models import create_fla_model
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False


AVAILABLE_ARCHITECTURES = {
    # Classic RNNs
    'lstm': create_lstm_model,
    'gru': create_gru_model,
    
    # Transformers
    'vanilla_transformer': create_vanilla_transformer,
    'rope': create_rope_model,
}

# Add FLA models if available
if FLA_AVAILABLE:
    AVAILABLE_ARCHITECTURES.update({
        'deltanet': lambda cfg: create_fla_model(cfg, 'deltanet'),
        'gated_deltanet': lambda cfg: create_fla_model(cfg, 'gated_deltanet'),
        'rwkv7': lambda cfg: create_fla_model(cfg, 'rwkv7'),
        'retention': lambda cfg: create_fla_model(cfg, 'retention'),
        'gla': lambda cfg: create_fla_model(cfg, 'gla'),
    })


def create_model(architecture: str, config: ModelConfig) -> BaseLanguageModel:
    """
    Create a model by architecture name
    
    Args:
        architecture: Name of architecture (lstm, gru, rope, deltanet, etc.)
        config: Model configuration
        
    Returns:
        Model instance
    """
    if architecture not in AVAILABLE_ARCHITECTURES:
        available = ', '.join(AVAILABLE_ARCHITECTURES.keys())
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: {available}"
        )
    
    config.architecture = architecture
    return AVAILABLE_ARCHITECTURES[architecture](config)


def list_available_architectures() -> list:
    """List all available architectures"""
    return list(AVAILABLE_ARCHITECTURES.keys())
