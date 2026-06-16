from typing import Any, Dict, List, Sequence, Tuple

import torch

from .beatmap import MAP_FEATURE_ATTRIBUTES
from .sampler import length_bucket


def rounded_pad_length(
    length: int, max_seq_len: int, buckets: Sequence[int] | None = None
) -> int:
    length = min(int(length), int(max_seq_len))
    if buckets:
        return min(length_bucket(length, buckets), int(max_seq_len))
    return min(((length + 127) // 128) * 128, int(max_seq_len))


def batch_vectors(
    vectors: Sequence[torch.Tensor],
    max_seq_len: int,
    vector_dim: int,
    *,
    pad_to_len: int | None = None,
    packed: bool = False,
) -> Dict[str, torch.Tensor | int]:
    effective_max_seq_len = (
        min(int(pad_to_len), int(max_seq_len))
        if pad_to_len is not None
        else int(max_seq_len)
    )
    lengths = [min(int(v.shape[0]), effective_max_seq_len) for v in vectors]
    seqlens = torch.tensor(lengths, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )

    if packed:
        packed_vectors = torch.zeros(sum(lengths), vector_dim, dtype=torch.float32)
        offset = 0
        for vector, length in zip(vectors, lengths):
            if length > 0:
                actual_dim = min(vector.shape[1], vector_dim)
                packed_vectors[offset : offset + length, :actual_dim] = vector[
                    :length, :actual_dim
                ]
                offset += length
        return {
            "packed_vectors": packed_vectors,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max(lengths) if lengths else 0,
        }

    max_len = effective_max_seq_len if pad_to_len is not None else max(lengths, default=0)
    padded = torch.zeros(len(vectors), max_len, vector_dim, dtype=torch.float32)
    mask = torch.zeros(len(vectors), max_len, dtype=torch.bool)
    for i, (vector, length) in enumerate(zip(vectors, lengths)):
        if length > 0:
            actual_dim = min(vector.shape[1], vector_dim)
            padded[i, :length, :actual_dim] = vector[:length, :actual_dim]
            mask[i, :length] = True
    return {"vectors": padded, "attention_mask": mask, "cu_seqlens": cu_seqlens}


def stack_dicts(dict_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not dict_list:
        return {}
    return {
        k: torch.tensor([d[k] for d in dict_list], dtype=torch.float32)
        for k in dict_list[0]
    }


def stack_map_features(dict_list: List[Dict[str, Any]]) -> torch.Tensor:
    if not dict_list:
        return torch.empty(0, len(MAP_FEATURE_ATTRIBUTES), dtype=torch.float32)
    return torch.tensor(
        [
            [float(features.get(name, 0.0)) for name in MAP_FEATURE_ATTRIBUTES]
            for features in dict_list
        ],
        dtype=torch.float32,
    )


def _alignment_labels(
    map_features: Tuple[Dict[str, Any], ...],
    beatmap_ids: Tuple[int, ...],
    targets: Tuple[Dict[str, Any], ...],
):
    id_to_batch = {int(bid): i for i, bid in enumerate(beatmap_ids)}
    positive_weights = torch.zeros(len(targets), len(targets), dtype=torch.float32)
    anchor_weights = torch.tensor(
        [float(target.get("anchor_weight", 1.0)) for target in targets],
        dtype=torch.float32,
    )
    beatmapset_ids = torch.tensor(
        [int(target.get("beatmapset_id", -1)) for target in targets], dtype=torch.long
    )
    valid_sets = beatmapset_ids >= 0
    same_set = beatmapset_ids[:, None] == beatmapset_ids[None, :]
    song_keys = [str(target.get("song_key", "")) for target in targets]
    same_song = torch.zeros(len(targets), len(targets), dtype=torch.bool)
    parsed_song_keys = [set(key.split("|")) - {""} for key in song_keys]
    for i, left in enumerate(parsed_song_keys):
        if not left:
            continue
        for j, right in enumerate(parsed_song_keys):
            same_song[i, j] = bool(left & right)

    ignore_contrastive = (
        same_set & valid_sets[:, None] & valid_sets[None, :]
    ) | same_song
    ignore_contrastive.fill_diagonal_(False)

    for i, target in enumerate(targets):
        for bid in target.get("ignore_ids", []):
            j = id_to_batch.get(int(bid))
            if j is not None and j != i:
                ignore_contrastive[i, j] = True
        for bid, weight in zip(
            target.get("graph_positive_ids", []), target.get("graph_positive_weights", [])
        ):
            j = id_to_batch.get(int(bid))
            if j is not None and j != i:
                existing = float(positive_weights[i, j])
                new_weight = max(float(weight), 0.0)
                positive_weights[i, j] = 1.0 - (1.0 - existing) * (1.0 - new_weight)
                ignore_contrastive[i, j] = True

    return {
        "positive_weights": positive_weights,
        "ignore_contrastive": ignore_contrastive,
        "anchor_weights": anchor_weights,
        "map_features": stack_map_features(list(map_features)),
    }


def collate_pretrain(
    batch: List[Tuple[torch.Tensor, Dict[str, float]]],
    max_seq_len: int,
    vector_dim: int,
    length_buckets: Sequence[int] | None = None,
):
    vectors, attrs = zip(*batch)
    max_len = max(min(v.shape[0], max_seq_len) for v in vectors)
    vector_batch = batch_vectors(
        vectors,
        max_seq_len,
        vector_dim,
        pad_to_len=rounded_pad_length(max_len, max_seq_len, length_buckets),
    )
    return (
        vector_batch["vectors"],
        vector_batch["attention_mask"],
        stack_dicts(attrs),
        vector_batch["cu_seqlens"],
    )


def collate_align(
    batch: List[Tuple],
    max_seq_len: int,
    vector_dim: int,
    *,
    packed: bool = False,
    length_buckets: Sequence[int] | None = None,
):
    vectors, _, map_features, beatmap_ids, targets = zip(*batch)
    labels = _alignment_labels(map_features, beatmap_ids, targets)
    labels["use_contrastive"] = packed

    if packed:
        vector_batch = batch_vectors(vectors, max_seq_len, vector_dim, packed=True)
        return {
            **vector_batch,
            "max_seqlen": torch.tensor(vector_batch["max_seqlen"], dtype=torch.long),
            "labels": labels,
            "batch_size": len(batch),
        }

    max_len = max(min(v.shape[0], max_seq_len) for v in vectors)
    vector_batch = batch_vectors(
        vectors,
        max_seq_len,
        vector_dim,
        pad_to_len=rounded_pad_length(max_len, max_seq_len, length_buckets),
    )
    return {**vector_batch, "labels": labels, "batch_size": len(batch)}
