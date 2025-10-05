"""Benchmark tasks for architecture comparison"""

from .mqar import MQARDataset, MQARConfig
from .flip_flop import FlipFlopDataset, FlipFlopConfig
from .tropical_tasks import TropicalTaskConfig, KnapsackTask
from .hidden_mode_task import HiddenModeConfig, HiddenModeDataset

__all__ = [
    'MQARDataset', 'MQARConfig',
    'FlipFlopDataset', 'FlipFlopConfig',
    'TropicalTaskConfig', 'KnapsackTask',
    'HiddenModeConfig', 'HiddenModeDataset'
]
