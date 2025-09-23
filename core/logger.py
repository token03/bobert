# logger.py
import numpy as np
import torch
from typing import Dict, List, Tuple 
from core.data.transforms import BeatmapNormalizer
from core.data.types import BeatmapMetadata, HitObjectVector, NormalizationType


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