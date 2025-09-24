# logger.py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

from core.data.transforms import BeatmapNormalizer
from core.data.types import BeatmapMetadata, HitObjectVector, NormalizationType
from core.model.bert import BertEncoder, BertForMaskedModeling

def log_model_summary(model: nn.Module):
    is_mlm_model = False

    actual_model = model
    if hasattr(model, '_orig_mod'):
        actual_model = model._orig_mod

    if isinstance(actual_model, BertForMaskedModeling):
        core_model = actual_model.bert
        is_mlm_model = True
    elif isinstance(actual_model, BertEncoder):
        core_model = actual_model
    else:
        raise ValueError(f"Unsupported model type: {type(actual_model)}. Expected BertForMaskedModeling or BertEncoder.")

    summary = core_model.get_summary()
    
    print(f"\n--- BERT Encoder Information ---")
    print(f"Total Parameters: {summary['trainable_parameters'] / 1e6:.2f}M")
    print(f"Model Dimension: {core_model.d_model}")
    print(f"Number of Heads: {core_model.n_heads}")
    print(f"Number of Layers: {core_model.n_layers}")
    print(f"Flash Attention: {core_model.use_flash_attention}")
    print("-" * 30)

    if is_mlm_model and isinstance(actual_model, BertForMaskedModeling):
        print(f"\n--- MLM Head Information ---")
        print(f"Task: Masked Modeling")
        print(f"Masking Ratio: {actual_model.masking_ratio}")
        print(f"Model Compiled: {actual_model.is_compiled}")
        print("-" * 30)


class TrainingLogger:
    def __init__(
        self,
        standard_cont_names: List[str],
        angle_pair_names: List[str],
        cat_feat_names: List[str]
    ):
        self.standard_cont_names = standard_cont_names
        self.angle_pair_names = angle_pair_names
        self.cat_feat_names = cat_feat_names

    def log_training_start(self, start_epoch: int, num_epochs: int, config: Dict[str, Any]):
        """Prints the header before training begins."""
        print(f"\n--- Starting Training ---")
        print(f"Epochs: {start_epoch + 1} to {num_epochs}")
        print(f"Batch Size: {config['training']['batch_size']}")
        print(f"Learning Rate: {config['training']['learning_rate']}")
        print("-" * 60)

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
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"LR: {train_metrics['learning_rate']:.2e} | "
            f"Time: {duration:.2f}s"
        )
        
        self._log_validation_details(val_metrics)

        print("-" * 70)
        print(f"Checkpoint saved to {checkpoint_path}")
        print("=" * 70)

    def _log_validation_details(self, val_metrics: Dict[str, Any]):
        """Prints the detailed validation report tables."""
        print("=" * 70)
        print(f"{' ' * 21} DETAILED VALIDATION REPORT {' ' * 22}")

        if 'continuous_metrics' in val_metrics:
            print("-" * 70)
            print(" CONTINUOUS FEATURES:")
            header = f"  {'Feature':<22} | {'MAE / Error':<15} | {'Mean (True)':<12} | {'Std (True)':<12}"
            print(header)
            print(f"  {'-'*22}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}")
            for name in self.standard_cont_names:
                 if name in val_metrics['continuous_metrics']:
                    metrics = val_metrics['continuous_metrics'][name]
                    row = f"  {name:<22} | {metrics['mae']:<15.4f} | {metrics['mean']:<12.4f} | {metrics['std']:<12.4f}"
                    print(row)
            for name in self.angle_pair_names:
                if name in val_metrics['continuous_metrics']:
                    metrics = val_metrics['continuous_metrics'][name]
                    row = f"  {name:<22} | {metrics['mae_degrees']:<15.4f} (deg) | {'-':<12} | {'-':<12}"
                    print(row)

        if 'categorical_metrics' in val_metrics:
            print("-" * 70)
            print(" CATEGORICAL FEATURES:")
            header = f"  {'Feature':<22} | {'Accuracy':<10} | {'Precision':<12} | {'Recall':<12}"
            print(header)
            print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
            for name, metrics in val_metrics['categorical_metrics'].items():
                row = f"  {name:<22} | {metrics['accuracy']:<10.2%} | {metrics['precision']:<12.4f} | {metrics['recall']:<12.4f}"
                print(row)
                dist_data = metrics['distribution']
                total_count = sum(dist_data.values())
                if total_count == 0: continue
                sorted_dist = sorted(dist_data.items(), key=lambda item: item[1], reverse=True)
                dist_str_parts, limit = [], 5
                if len(sorted_dist) > limit:
                    top_items = sorted_dist[:limit]
                    other_count = sum(count for _, count in sorted_dist[limit:])
                    for class_idx, count in top_items: dist_str_parts.append(f"{class_idx}:{count/total_count:.1%}")
                    if other_count > 0: dist_str_parts.append(f"Other:{other_count/total_count:.1%}")
                else:
                    for class_idx, count in sorted_dist: dist_str_parts.append(f"{class_idx}:{count/total_count:.1%}")
                print(f"    └─ True Dist: {', '.join(dist_str_parts)}")
    
    def log_training_end(self):
        """Prints the footer when training is finished."""
        print("\n--- Training Finished ---")


def _print_stats_table(title: str, field_names: List[str], norm_specs: Dict, descriptions: Dict, stats: Dict):
    print(f"\n--- {title}:")
    print("-" * 70)
    print(f"{'Field Name':<20} {'Type':<12} {'Param 1':<12} {'Param 2':<12} {'Description'}")
    print("-" * 70)

    for field_name in field_names:
        norm_type = norm_specs[field_name]
        description = descriptions.get(field_name, 'Unknown field')
        param1_str, param2_str = "N/A", "N.A."
        type_str = str(norm_type.value)

        if norm_type == NormalizationType.CATEGORICAL:
            type_str = "categorical"
        elif field_name in stats:
            param1, param2 = stats[field_name]
            param1_str = f"{param1:.4f}"
            param2_str = f"{param2:.4f}"
            if norm_type == NormalizationType.STANDARD:
                type_str = "mean/std"
            elif norm_type == NormalizationType.LOG:
                type_str = "log+norm"
            elif norm_type == NormalizationType.MINMAX:
                type_str = "min/max"

        print(f"{field_name:<20} {type_str:<12} {param1_str:<12} {param2_str:<12} {description}")


def calculate_normalization_stats(
    train_data: List[Tuple[torch.Tensor, torch.Tensor]],
    include_augmentation: bool = True
) -> BeatmapNormalizer:
    normalizer = BeatmapNormalizer.from_data(train_data, include_augmentation)

    print("\n" + "="*70)
    print("                    NORMALIZATION STATISTICS")
    print("="*70)

    _print_stats_table(
        "VECTOR STATISTICS",
        HitObjectVector.get_field_names(),
        HitObjectVector.get_normalization_specs(),
        HitObjectVector.get_field_descriptions(),
        normalizer.get_vector_stats()
    )

    _print_stats_table(
        "METADATA STATISTICS",
        BeatmapMetadata.get_field_names(),
        BeatmapMetadata.get_normalization_specs(),
        BeatmapMetadata.get_field_descriptions(),
        normalizer.get_metadata_stats()
    )

    print("="*70)
    return normalizer

def print_data_summary(all_data: List[Tuple[torch.Tensor, torch.Tensor]]):
    if not all_data:
        print("No data loaded!")
        return

    total_maps = len(all_data)
    metadata_dim = all_data[0][1].shape[0]
    vector_dim = all_data[0][0].shape[1]

    seq_lengths = [data[0].shape[0] for data in all_data]
    avg_seq_len = np.mean(seq_lengths)
    max_seq_len = np.max(seq_lengths)
    min_seq_len = np.min(seq_lengths)

    print(f"\n--- Data Summary ---")
    print(f"Total beatmaps: {total_maps}")
    print(f"Vector dimension: {vector_dim}")
    print(f"Metadata dimension: {metadata_dim}")
    print(f"Sequence length - Min: {min_seq_len}, Max: {max_seq_len}, Avg: {avg_seq_len:.1f}")
    print("-" * 20)