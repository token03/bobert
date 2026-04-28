from typing import Any, Dict, List, Tuple, Union

import torch


def random_split_aligned(
    data_sources: Dict[str, Union[List, Dict]], val_split: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    primary_key = next(iter(data_sources))
    total_len = len(data_sources[primary_key])

    val_size = int(total_len * val_split)
    train_size = total_len - val_size

    indices = torch.randperm(total_len).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    def extract(source, idx_list):
        if isinstance(source, dict):
            return {k: [v[i] for i in idx_list] for k, v in source.items()}
        return [source[i] for i in idx_list]

    train_out = {k: extract(v, train_idx) for k, v in data_sources.items()}
    val_out = {k: extract(v, val_idx) for k, v in data_sources.items()}
    return train_out, val_out
