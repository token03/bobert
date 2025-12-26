# logger.py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from core.model.bobert import Bobert, BobertForPretraining

def log_model_summary(model: nn.Module):
    is_pretrain_model = False

    actual_model = model
    if hasattr(model, '_orig_mod'):
        actual_model = model._orig_mod

    if isinstance(actual_model, BobertForPretraining):
        core_model = actual_model.bert
        is_pretrain_model = True
    elif isinstance(actual_model, Bobert):
        core_model = actual_model
    else:
        try:
            core_model = actual_model.bert if hasattr(actual_model, 'bert') else actual_model
        except:
             raise ValueError(f"Unsupported model type: {type(actual_model)}.")


    summary = core_model.get_summary()
    
    print(f"\n--- BERT Encoder Information ---")
    print(f"Total Parameters: {summary['trainable_parameters'] / 1e6:.2f}M")
    print(f"Model Dimension: {core_model.d_model}")
    print(f"Number of Heads: {core_model.n_heads}")
    print(f"Number of Layers: {core_model.n_layers}")
    print(f"Flash Attention: {core_model.use_flash_attention}")
    print("-" * 30)

    if is_pretrain_model and isinstance(actual_model, BobertForPretraining):
        print(f"\n--- Pre-training Head Information ---")
        print(f"Tasks: Masked Modeling, Difficulty Attribute Prediction")
        print(f"Masking Ratio: {actual_model.masking_ratio}")
        print(f"Model Compiled: {actual_model.is_compiled}")
        print("-" * 30)


class TrainingLogger:
    def __init__(
        self,
        standard_cont_names: List[str],
        cat_feat_names: List[str]
    ):
        self.standard_cont_names = standard_cont_names
        self.cat_feat_names = cat_feat_names

    def log_training_start(self, start_epoch: int, num_epochs: int, config: Dict[str, Any]):
        """Prints the header before training begins."""
        print(f"\n--- Starting Training ---")
        print(f"Epochs: {start_epoch + 1} to {num_epochs}")
        print(f"Batch Size: {config['pretraining']['batch_size']}")
        print(f"Learning Rate: {config['pretraining']['learning_rate']}")
        print("-" * 30)

    def log_epoch_end(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, Any],
        duration: float,
        checkpoint_path: str
    ):
        """Logs all information at the end of an epoch."""
        train_loss = train_metrics.get('total_loss', train_metrics.get('loss', 0.0))
        val_loss = val_metrics.get('total_loss', val_metrics.get('loss', 0.0))
        
        train_mlm = train_metrics.get('mlm_loss', 0.0)
        val_mlm = val_metrics.get('mlm_loss', 0.0)
        train_diff = train_metrics.get('difficulty_loss', 0.0)
        val_diff = val_metrics.get('difficulty_loss', 0.0)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} (MLM: {train_mlm:.4f}, Diff: {train_diff:.4f}) | "
            f"Val Loss: {val_loss:.4f} (MLM: {val_mlm:.4f}, Diff: {val_diff:.4f}) | "
            f"LR: {train_metrics['learning_rate']:.2e} | "
            f"Time: {duration:.2f}s"
        )
        
        self._log_validation_details(val_metrics)

        print("-" * 30)
        print(f"Checkpoint saved to {checkpoint_path}")
        print("-" * 30)

    def _log_validation_details(self, val_metrics: Dict[str, Any]):
        """Prints the detailed validation report tables."""
        print("-" * 30)
        print(f"Validation Metrics")

        if 'difficulty_metrics' in val_metrics:
            print("Difficulty MAE:")
            for name, mae in val_metrics['difficulty_metrics'].items():
                attr_name = name.replace('_mae', '')
                print(f"  {attr_name:<15}: {mae:.4f}")

        if 'continuous_metrics' in val_metrics:
            print("Continuous Features:")
            for name in self.standard_cont_names:
                 if name in val_metrics['continuous_metrics']:
                    metrics = val_metrics['continuous_metrics'][name]
                    print(f"  {name:<15}: MAE {metrics['mae']:.4f}")

        if 'categorical_metrics' in val_metrics:
            print("Categorical Features:")
            for name, metrics in val_metrics['categorical_metrics'].items():
                print(f"  {name:<15}: Acc {metrics['accuracy']:.2%}, Prec {metrics['precision']:.4f}, Rec {metrics['recall']:.4f}")
    
    def log_training_end(self):
        print("\n--- Training Finished ---")

def print_data_summary(all_data: List[Any]):
    if not all_data:
        print("No data loaded!")
        return

    total_maps = len(all_data)

    def _extract_vectors(item: Any) -> torch.Tensor:
        if isinstance(item, tuple):
            return item[0]
        return item

    vectors_sample = _extract_vectors(all_data[0])
    vector_dim = vectors_sample.shape[1]

    seq_lengths = [_extract_vectors(data).shape[0] for data in all_data]
    avg_seq_len = np.mean(seq_lengths)
    max_seq_len = np.max(seq_lengths)
    min_seq_len = np.min(seq_lengths)

    print(f"\n--- Data Summary ---")
    print(f"Total beatmaps: {total_maps}")
    print(f"Vector dimension: {vector_dim}")
    print(f"Sequence length - Min: {min_seq_len}, Max: {max_seq_len}, Avg: {avg_seq_len:.1f}")
    print("-" * 20)