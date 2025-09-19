# bert_utils.py
import torch
import torch.nn as nn
from typing import Dict, Any

from .bert import BertEncoder, BertForMaskedModeling
from ..data.types import VECTOR_DIM

def create_bert_encoder(config: Dict[str, Any], device: torch.device) -> BertEncoder:
    """Create a BERT encoder directly from config."""
    model_config = config['model']
    components_config = config.get('components', {})
    
    attention_type = 'rope' if components_config.get('use_rope', True) else 'standard'
    norm_type = components_config.get('norm_type', 'rmsnorm')
    ffn_type = components_config.get('ffn_type', 'swiglu')
    
    dim_feedforward = model_config['d_model'] * model_config.get('dim_feedforward_mult', 4)
    
    model = BertEncoder(
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        dim_feedforward=dim_feedforward,
        dropout=model_config.get('dropout', 0.1),
        metadata_dim=5,  
        attention_type=attention_type,
        norm_type=norm_type,
        ffn_type=ffn_type
    )
    
    return model.to(device)

def create_bert_for_mlm(config: Dict[str, Any], device: torch.device) -> BertForMaskedModeling:
    """Create a BERT model for masked language modeling."""
    base_model = create_bert_encoder(config, device)
    
    masking_ratio = config.get('mlm', {}).get('masking_ratio', 0.15)
    model = BertForMaskedModeling(base_model, masking_ratio)
    
    model = model.to(device)
    
    components_config = config.get('components', {})
    if components_config.get('compile_model', False):
        print("Compiling BERT model with torch.compile...")
        model = torch.compile(model, mode="default")
    
    return model

def print_bert_info(model: nn.Module, config: Dict[str, Any]):
    """Print information about the BERT model."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n--- BERT Model Information ---")
    print(f"Total Parameters: {num_params / 1e6:.2f}M")
    print(f"Model Dimension: {config['model']['d_model']}")
    print(f"Number of Heads: {config['model']['n_heads']}")
    print(f"Number of Layers: {config['model']['n_layers']}")
    
    components = config.get('components', {})
    print(f"RoPE Enabled: {components.get('use_rope', True)}")
    print(f"Flash Attention: {components.get('use_flash_attention', True)}")
    print(f"Model Compiled: {components.get('compile_model', False)}")
    print("-" * 30)

def get_bert_summary(model: nn.Module) -> Dict[str, Any]:
    """Get a summary of BERT model statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  
        'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0
    }