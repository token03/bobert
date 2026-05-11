from typing import Any, Dict, List, Sequence, Tuple

import torch

from .beatmap import MAP_FEATURE_ATTRIBUTES


def pad_batch(
    vectors: List[torch.Tensor],
    max_seq_len: int,
    vector_dim: int,
    pad_to_len: int | None = None,
):
    effective_max_seq_len = (
        min(int(pad_to_len), max_seq_len) if pad_to_len is not None else max_seq_len
    )
    lengths = [min(v.shape[0], effective_max_seq_len) for v in vectors]
    if pad_to_len is None:
        max_len = max(lengths) if lengths else 0
    else:
        max_len = effective_max_seq_len
    batch_size = len(vectors)

    padded = torch.zeros(batch_size, max_len, vector_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (v, length) in enumerate(zip(vectors, lengths)):
        if length > 0:
            actual_dim = min(v.shape[1], vector_dim)
            padded[i, :length, :actual_dim] = v[:length, :actual_dim]
            mask[i, :length] = True

    seqlens = torch.tensor(lengths, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    return padded, mask, cu_seqlens


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


def _length_bucket(length: int, buckets: Sequence[int]) -> int:
    for bucket in buckets:
        if length <= bucket:
            return int(bucket)
    return int(buckets[-1])


def _alignment_labels(
    attrs: Tuple[Dict[str, Any], ...],
    map_features: Tuple[Dict[str, Any], ...],
    beatmap_ids: Tuple[int, ...],
    targets: Tuple[Dict[str, Any], ...],
    ignore_near_star_delta: float,
):
    teacher_dim = 0
    for target in targets:
        teacher_dim = max(teacher_dim, len(target.get("graph_embedding", [])))
    teacher_dim = teacher_dim or 128

    graph_teacher = torch.zeros(len(targets), teacher_dim, dtype=torch.float32)
    has_teacher = torch.zeros(len(targets), dtype=torch.bool)
    for i, target in enumerate(targets):
        teacher = target.get("graph_embedding", [])
        if teacher:
            teacher_tensor = torch.tensor(teacher[:teacher_dim], dtype=torch.float32)
            graph_teacher[i, : teacher_tensor.shape[0]] = teacher_tensor
            has_teacher[i] = True

    id_to_batch = {int(bid): i for i, bid in enumerate(beatmap_ids)}
    positive_weights = torch.zeros(len(targets), len(targets), dtype=torch.float32)
    beatmapset_ids = torch.tensor(
        [int(target.get("beatmapset_id", -1)) for target in targets], dtype=torch.long
    )
    stars = torch.tensor(
        [float(target.get("stars", float("nan"))) for target in targets],
        dtype=torch.float32,
    )
    valid_sets = beatmapset_ids >= 0
    valid_stars = ~torch.isnan(stars)
    same_set = beatmapset_ids[:, None] == beatmapset_ids[None, :]
    near_star = torch.abs(stars[:, None] - stars[None, :]) <= ignore_near_star_delta
    song_keys = [str(target.get("song_key", "")) for target in targets]
    same_song = torch.zeros(len(targets), len(targets), dtype=torch.bool)
    parsed_song_keys = [set(key.split("|")) - {""} for key in song_keys]
    for i, left in enumerate(parsed_song_keys):
        if not left:
            continue
        for j, right in enumerate(parsed_song_keys):
            same_song[i, j] = bool(left & right)

    ignore_contrastive = near_star & valid_stars[:, None] & valid_stars[None, :]
    ignore_contrastive &= (
        same_set & valid_sets[:, None] & valid_sets[None, :]
    ) | same_song
    ignore_contrastive.fill_diagonal_(False)

    def combine_weight(existing: torch.Tensor, new_weight: float) -> float:
        existing_value = float(existing)
        new_weight = max(float(new_weight), 0.0)
        return 1.0 - (1.0 - existing_value) * (1.0 - new_weight)

    for i, target in enumerate(targets):
        for ids_key, weights_key in (
            ("graph_positive_ids", "graph_positive_weights"),
            ("song_positive_ids", "song_positive_weights"),
            ("creator_positive_ids", "creator_positive_weights"),
            ("cross_status_positive_ids", "cross_status_positive_weights"),
        ):
            for bid, weight in zip(target.get(ids_key, []), target.get(weights_key, [])):
                j = id_to_batch.get(int(bid))
                if j is not None and j != i:
                    positive_weights[i, j] = combine_weight(positive_weights[i, j], weight)
                    ignore_contrastive[i, j] = True

    return {
        "graph_teacher": graph_teacher,
        "has_teacher": has_teacher,
        "positive_weights": positive_weights,
        "ignore_contrastive": ignore_contrastive,
        "difficulty": stack_dicts(list(attrs)),
        "map_features": stack_map_features(list(map_features)),
    }


def collate_pretrain(
    batch: List[Tuple[torch.Tensor, Dict[str, float]]],
    max_seq_len: int,
    vector_dim: int,
):
    vectors, attrs = zip(*batch)
    padded, mask, cu_seqlens = pad_batch(vectors, max_seq_len, vector_dim)
    return padded, mask, stack_dicts(attrs), cu_seqlens


def collate_align(
    batch: List[Tuple],
    max_seq_len: int,
    vector_dim: int,
    ignore_near_star_delta: float,
):
    vectors, attrs, map_features, beatmap_ids, targets = zip(*batch)
    padded_vec, mask, cu_seqlens = pad_batch(vectors, max_seq_len, vector_dim)
    labels = _alignment_labels(
        attrs, map_features, beatmap_ids, targets, ignore_near_star_delta
    )

    return (
        padded_vec,
        mask,
        cu_seqlens,
        labels["graph_teacher"],
        labels["has_teacher"],
        labels["positive_weights"],
        labels["ignore_contrastive"],
        labels["difficulty"],
        labels["map_features"],
    )


def collate_align_chunked(
    batch: List[Tuple],
    max_seq_len: int,
    vector_dim: int,
    group_size: int,
    forward_length_buckets: Sequence[int],
    max_forward_tokens: int,
    ignore_near_star_delta: float,
):
    vectors, attrs, map_features, beatmap_ids, targets = zip(*batch)
    labels = _alignment_labels(
        attrs, map_features, beatmap_ids, targets, ignore_near_star_delta
    )

    buckets = sorted(int(bucket) for bucket in forward_length_buckets)
    if not buckets:
        raise ValueError("forward_length_buckets must not be empty")
    if buckets[-1] < max_seq_len:
        raise ValueError("forward_length_buckets must cover max_seq_len")
    max_forward_tokens = int(max_forward_tokens)
    if max_forward_tokens <= 0:
        raise ValueError("max_forward_tokens must be positive")

    max_forward_tokens = min(max_forward_tokens, 1024 * 32)

    chunk_groups: Dict[int, List[int]] = {bucket: [] for bucket in buckets}
    for start in range(0, len(vectors), group_size):
        positions = list(range(start, min(start + group_size, len(vectors))))
        group_max_len = max(min(vectors[pos].shape[0], max_seq_len) for pos in positions)
        bucket = _length_bucket(group_max_len, buckets)
        chunk_groups[bucket].extend(positions)

    chunks = []
    for bucket in buckets:
        positions = chunk_groups[bucket]
        if not positions:
            continue

        samples_per_chunk = max_forward_tokens // bucket
        if samples_per_chunk < 1:
            raise ValueError("max_forward_tokens must fit at least one bucketed sample")

        for chunk_start in range(0, len(positions), samples_per_chunk):
            chunk_positions = positions[chunk_start : chunk_start + samples_per_chunk]
            chunk_vectors = [vectors[pos] for pos in chunk_positions]
            padded_vec, mask, cu_seqlens = pad_batch(
                chunk_vectors,
                max_seq_len,
                vector_dim,
                pad_to_len=bucket,
            )
            chunks.append(
                {
                    "vectors": padded_vec,
                    "attention_mask": mask,
                    "cu_seqlens": cu_seqlens,
                    "positions": torch.tensor(chunk_positions, dtype=torch.long),
                    "bucket": torch.tensor(bucket, dtype=torch.long),
                }
            )

    labels["use_contrastive"] = True
    return {
        "chunks": chunks,
        "labels": labels,
        "batch_size": len(batch),
    }
