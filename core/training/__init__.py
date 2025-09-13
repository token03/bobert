"""Core training utilities."""

from .trainer import (
    MLMTrainer,
    CheckpointManager,
    MetricsTracker,
    setup_training,
    create_optimizer,
    create_scheduler,
    mlm_loss_fn
)

def safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert value to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

__all__ = [
    'MLMTrainer',
    'CheckpointManager', 
    'MetricsTracker',
    'setup_training',
    'create_optimizer',
    'create_scheduler',
    'mlm_loss_fn',
    'safe_float',
    'safe_int'
]