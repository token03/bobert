from typing import Any, Dict, List, Tuple

import torch


def pad_batch(vectors: List[torch.Tensor], max_seq_len: int, vector_dim: int):
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
    max_tags: int = 50,
):
    vectors, metadata, tags, attrs, beatmap_ids, targets = zip(*batch)
    padded_vec, mask, cu_seqlens = pad_batch(vectors, max_seq_len, vector_dim)

    tag_lens = [min(t.shape[0], max_tags) for t in tags]
    padded_tags = torch.zeros(
        len(batch), max(tag_lens) if tag_lens else 1, dtype=torch.long
    )
    for i, (t, length) in enumerate(zip(tags, tag_lens)):
        if length > 0:
            padded_tags[i, :length] = t[:length]

    teacher_dim = 0
    for target in targets:
        teacher_dim = max(teacher_dim, len(target.get("lgcn_embedding", [])))
    teacher_dim = teacher_dim or 64

    lgcn_teacher = torch.zeros(len(batch), teacher_dim, dtype=torch.float32)
    status_labels = torch.zeros(len(batch), dtype=torch.float32)
    has_teacher = torch.zeros(len(batch), dtype=torch.bool)
    for i, target in enumerate(targets):
        teacher = target.get("lgcn_embedding", [])
        if teacher:
            teacher_tensor = torch.tensor(teacher[:teacher_dim], dtype=torch.float32)
            lgcn_teacher[i, : teacher_tensor.shape[0]] = teacher_tensor
            has_teacher[i] = True
        status_labels[i] = 1.0 if target.get("status_group") == "ranked" else 0.0

    return (
        padded_vec,
        mask,
        cu_seqlens,
        torch.tensor(beatmap_ids, dtype=torch.long),
        lgcn_teacher,
        has_teacher,
        status_labels,
        padded_tags,
        stack_dicts(attrs),
    )
