import torch
import torch.nn as nn
from typing import Dict, Any, Type, Union

from .bert import BertEncoder, BertForMaskedModeling
from .components import (
    create_attention_layer,
    create_norm_layer,
    create_ffn_layer
)

class ModelRegistry:
    _models: Dict[str, Type[nn.Module]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(model_class: Type[nn.Module]):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[nn.Module]:
        if name not in cls._models:
            raise ValueError(f"Unknown model type: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._models.keys())

ModelRegistry.register("bert")(BertEncoder)

def create_model_from_config(config: Dict[str, Any], device: torch.device) -> nn.Module:
    model_config = config['model']
    data_config = config['data']
    components_config = config.get('components', {})
    
    model_type = model_config['type']
    model_class = ModelRegistry.get_model_class(model_type)
    
    attention_type = 'rope' if components_config.get('use_rope', True) else 'standard'
    norm_type = components_config.get('norm_type', 'rmsnorm')
    ffn_type = components_config.get('ffn_type', 'swiglu')
    
    dim_feedforward = model_config['d_model'] * model_config.get('dim_feedforward_mult', 4)
    
    common_args = {
        'max_seq_len': data_config['max_seq_len'],
        'd_model': model_config['d_model'],
        'n_heads': model_config['n_heads'],
        'n_layers': model_config['n_layers'],
        'dim_feedforward': dim_feedforward,
        'dropout': model_config.get('dropout', 0.1),
        'in_channels': data_config['in_channels'],
        'metadata_dim': 5,  
        'attention_type': attention_type,
        'norm_type': norm_type,
        'ffn_type': ffn_type
    }
    
    if model_type == 'bert':
        model = model_class(**common_args)
    else:
        raise ValueError(f"Model creation not implemented for type: {model_type}")
    
    model = model.to(device)
    
    if components_config.get('compile_model', False):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    return model

def create_task_model_from_config(
    config: Dict[str, Any], 
    device: torch.device,
    task_type: str = 'mlm'
) -> nn.Module:
    base_model = create_model_from_config(config, device)
    
    if task_type == 'mlm':
        masking_ratio = config.get('mlm', {}).get('masking_ratio', 0.15)
        model = BertForMaskedModeling(
            base_model, 
            config['data']['in_channels'], 
            masking_ratio
        )
    else:
        raise ValueError(f"Task type not implemented: {task_type}")
    
    model = model.to(device)
    
    components_config = config.get('components', {})
    if components_config.get('compile_model', False):
        print("Compiling task model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    return model

def print_model_info(model: nn.Module, config: Dict[str, Any]):
    """Prints information about the model."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n--- Model Information ---")
    print(f"Model Type: {config['model']['type']}")
    print(f"Total Parameters: {num_params / 1e6:.2f}M")
    print(f"Model Dimension: {config['model']['d_model']}")
    print(f"Number of Heads: {config['model']['n_heads']}")
    print(f"Number of Layers: {config['model']['n_layers']}")
    
    components = config.get('components', {})
    print(f"RoPE Enabled: {components.get('use_rope', True)}")
    print(f"Flash Attention: {components.get('use_flash_attention', True)}")
    print(f"Model Compiled: {components.get('compile_model', False)}")
    print("-" * 25)

def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Gets a summary of model statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  
        'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0
    }