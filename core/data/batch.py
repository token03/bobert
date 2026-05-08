from typing import Any, Dict, List, Tuple

import torch


def pad_batch(
    vectors: List[torch.Tensor],
    max_seq_len: int,
    vector_dim: int,
):
    lengths = [min(v.shape[0], max_seq_len) for v in vectors]
    max_len = max(lengths) if lengths else 0
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
):
    vectors, attrs, beatmap_ids, targets = zip(*batch)
    padded_vec, mask, cu_seqlens = pad_batch(vectors, max_seq_len, vector_dim)

    teacher_dim = 0
    for target in targets:
        teacher_dim = max(teacher_dim, len(target.get("graph_embedding", [])))
    teacher_dim = teacher_dim or 128

    graph_teacher = torch.zeros(len(batch), teacher_dim, dtype=torch.float32)
    has_teacher = torch.zeros(len(batch), dtype=torch.bool)
    for i, target in enumerate(targets):
        teacher = target.get("graph_embedding", [])
        if teacher:
            teacher_tensor = torch.tensor(teacher[:teacher_dim], dtype=torch.float32)
            graph_teacher[i, : teacher_tensor.shape[0]] = teacher_tensor
            has_teacher[i] = True

    id_to_batch = {int(bid): i for i, bid in enumerate(beatmap_ids)}
    positive_weights = torch.zeros(len(batch), len(batch), dtype=torch.float32)
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
    near_star = torch.abs(stars[:, None] - stars[None, :]) <= 0.01
    ignore_contrastive = (
        same_set
        & near_star
        & valid_sets[:, None]
        & valid_sets[None, :]
        & valid_stars[:, None]
        & valid_stars[None, :]
    )
    ignore_contrastive.fill_diagonal_(False)
    for i, target in enumerate(targets):
        for ids_key, weights_key in (
            ("positive_ids", "positive_weights"),
            ("cross_status_positive_ids", "cross_status_positive_weights"),
        ):
            for bid, weight in zip(target.get(ids_key, []), target.get(weights_key, [])):
                j = id_to_batch.get(int(bid))
                if j is not None and j != i:
                    positive_weights[i, j] = max(
                        float(positive_weights[i, j]), float(weight)
                    )

    return (
        padded_vec,
        mask,
        cu_seqlens,
        graph_teacher,
        has_teacher,
        positive_weights,
        ignore_contrastive,
        stack_dicts(attrs),
    )
