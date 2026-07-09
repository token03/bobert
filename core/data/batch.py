import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

from .schema import MAP_FEATURE_ATTRIBUTES


class LengthBucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        max_tokens: int,
        seed: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        return len(self._batches())

    def _batches(self) -> List[List[int]]:
        indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches = []
        batch: List[int] = []
        max_len = 0
        for idx in indices:
            length = self.lengths[idx]
            next_max_len = max(max_len, length)
            if batch and (
                len(batch) == self.batch_size
                or next_max_len * (len(batch) + 1) > self.max_tokens
            ):
                batches.append(batch)
                batch = []
                max_len = 0

            batch.append(idx)
            max_len = max(max_len, length)

        if batch and not self.drop_last:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._batches()
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        yield from batches


class AlignmentBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        beatmap_ids: List[int],
        mining_lookup: Dict[int, Dict[str, Any]],
        batch_size: int,
        anchor_indices: Sequence[int],
        epoch_size: int,
        group_size: int,
        seed: int,
        drop_last: bool,
    ):
        if batch_size % group_size != 0:
            raise ValueError("alignment batch_size must be divisible by group_size")
        if epoch_size <= 0:
            raise ValueError("alignment epoch_size must be positive")

        self.beatmap_ids = [int(x) for x in beatmap_ids]
        self.mining_lookup = mining_lookup
        self.batch_size = batch_size
        self.group_size = group_size
        self.seed = seed
        self.id_to_idx = {bid: i for i, bid in enumerate(self.beatmap_ids)}
        self.anchor_indices = [int(idx) for idx in anchor_indices]
        self.epoch_size = int(epoch_size)
        self.groups_per_batch = batch_size // group_size
        self.epoch = 0
        self.drop_last = drop_last

    def __len__(self) -> int:
        anchor_count = self._anchor_count()
        if self.drop_last:
            return anchor_count // self.groups_per_batch
        return math.ceil(anchor_count / self.groups_per_batch)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _anchor_count(self) -> int:
        return min(self.epoch_size, len(self.anchor_indices))

    def _sample_positives(
        self, mining: Dict[str, Any], rng: random.Random, count: int
    ) -> List[int]:
        ids = [int(bid) for bid in mining["graph_positive_ids"]]
        weights = [float(w) for w in mining["graph_positive_weights"]]
        candidates = [
            (bid, max(weight, 0.0))
            for bid, weight in zip(ids, weights)
            if bid in self.id_to_idx
        ]
        if len(candidates) < count:
            return []

        selected = []
        for _ in range(count):
            total = sum(weight for _, weight in candidates)
            if total <= 0.0:
                index = rng.randrange(len(candidates))
            else:
                threshold = rng.random() * total
                cumulative = 0.0
                index = len(candidates) - 1
                for i, (_, weight) in enumerate(candidates):
                    cumulative += weight
                    if cumulative >= threshold:
                        index = i
                        break
            bid, _ = candidates.pop(index)
            selected.append(bid)
        return selected

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        anchor_indices = self.anchor_indices.copy()
        rng.shuffle(anchor_indices)
        anchor_indices = anchor_indices[: self._anchor_count()]

        batch: List[int] = []
        for anchor_idx in anchor_indices:
            anchor_id = self.beatmap_ids[anchor_idx]
            mining = self.mining_lookup[anchor_id]
            positive_ids = self._sample_positives(mining, rng, self.group_size - 1)
            if not positive_ids:
                continue

            batch.extend([anchor_idx, *(self.id_to_idx[bid] for bid in positive_ids)])
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch


def rounded_pad_length(length: int, max_seq_len: int, buckets: Sequence[int]) -> int:
    length = min(int(length), int(max_seq_len))
    for bucket in buckets:
        if length <= bucket:
            return min(int(bucket), int(max_seq_len))
    return min(((length + 127) // 128) * 128, int(max_seq_len))


def _batch_lengths(
    vectors: Sequence[torch.Tensor],
    max_seq_len: int,
    *,
    pad_to_len: int,
) -> tuple[int, list[int], torch.Tensor]:
    effective_max_seq_len = min(int(pad_to_len), int(max_seq_len))
    lengths = [min(int(v.shape[0]), effective_max_seq_len) for v in vectors]
    seqlens = torch.tensor(lengths, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    return effective_max_seq_len, lengths, cu_seqlens


def batch_packed_vectors(
    vectors: Sequence[torch.Tensor],
    max_seq_len: int,
) -> Dict[str, torch.Tensor | int]:
    _, lengths, cu_seqlens = _batch_lengths(
        vectors,
        max_seq_len,
        pad_to_len=max_seq_len,
    )
    return {
        "packed_vectors": torch.cat(
            [vector[:length] for vector, length in zip(vectors, lengths)], dim=0
        ),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max(lengths),
    }


def batch_padded_vectors(
    vectors: Sequence[torch.Tensor],
    max_seq_len: int,
    *,
    pad_to_len: int,
) -> Dict[str, torch.Tensor]:
    effective_max_seq_len, lengths, cu_seqlens = _batch_lengths(
        vectors,
        max_seq_len,
        pad_to_len=pad_to_len,
    )

    padded = pad_sequence(
        [vector[:length] for vector, length in zip(vectors, lengths)], batch_first=True
    )
    padded = pad(padded, (0, 0, 0, effective_max_seq_len - padded.shape[1]))

    seqlens = torch.tensor(lengths, device=padded.device)
    mask = torch.arange(effective_max_seq_len, device=padded.device).unsqueeze(
        0
    ) < seqlens.unsqueeze(1)
    return {
        "vectors": padded,
        "attention_mask": mask,
        "cu_seqlens": cu_seqlens,
    }


def stack_dicts(dict_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        k: torch.tensor([d[k] for d in dict_list], dtype=torch.float32)
        for k in dict_list[0]
    }


def stack_map_features(dict_list: List[Dict[str, Any]]) -> torch.Tensor:
    return torch.tensor(
        [
            [float(features[name]) for name in MAP_FEATURE_ATTRIBUTES]
            for features in dict_list
        ],
        dtype=torch.float32,
    )


def pad_int_lists(
    values: Sequence[Sequence[int]], fill_value: int = -1
) -> torch.Tensor:
    return pad_sequence(
        [torch.tensor(value, dtype=torch.long) for value in values],
        batch_first=True,
        padding_value=fill_value,
    )


def pad_float_lists(values: Sequence[Sequence[float]]) -> torch.Tensor:
    return pad_sequence(
        [torch.tensor(value, dtype=torch.float32) for value in values], batch_first=True
    )


def _alignment_labels(
    map_features: Tuple[Dict[str, Any], ...] | None,
    beatmap_ids: Tuple[int, ...],
    targets: Tuple[Dict[str, Any], ...],
):
    labels = {
        "beatmap_ids": torch.tensor(beatmap_ids, dtype=torch.long),
        "beatmapset_ids": torch.tensor(
            [int(target["beatmapset_id"]) for target in targets], dtype=torch.long
        ),
        "song_ids": torch.tensor(
            [int(target["song_id"]) for target in targets], dtype=torch.long
        ),
        "graph_positive_ids": pad_int_lists(
            [target["graph_positive_ids"] for target in targets]
        ),
        "graph_positive_weights": pad_float_lists(
            [target["graph_positive_weights"] for target in targets]
        ),
        "ignore_ids": pad_int_lists([target["ignore_ids"] for target in targets]),
        "anchor_weights": torch.tensor(
            [float(target["anchor_weight"]) for target in targets],
            dtype=torch.float32,
        ),
    }
    if map_features is not None:
        labels["map_features"] = stack_map_features(list(map_features))
    return labels


def collate_pretrain(
    batch: List[Tuple[torch.Tensor, Dict[str, float]]],
    max_seq_len: int,
    length_buckets: Sequence[int],
    masking_ratio: float,
    mean_span_length: float,
):
    vectors, attrs = zip(*batch)
    max_len = max(min(v.shape[0], max_seq_len) for v in vectors)
    vector_batch = batch_padded_vectors(
        vectors,
        max_seq_len,
        pad_to_len=rounded_pad_length(max_len, max_seq_len, length_buckets),
    )
    return (
        vector_batch["vectors"],
        vector_batch["attention_mask"],
        stack_dicts(attrs),
        vector_batch["cu_seqlens"],
        generate_span_mask(
            vector_batch["attention_mask"],
            masking_ratio=masking_ratio,
            mean_span_length=mean_span_length,
        ),
    )


def generate_span_mask(
    attention_mask: torch.Tensor,
    *,
    masking_ratio: float,
    mean_span_length: float,
) -> torch.Tensor:
    batch_size, _ = attention_mask.shape
    device = attention_mask.device

    final_mask = torch.zeros_like(attention_mask)
    valid_lengths = attention_mask.sum(dim=1).long()
    target_counts = (valid_lengths.float() * float(masking_ratio)).round().long()
    max_target = int(target_counts.max().item()) if target_counts.numel() else 0
    if max_target == 0:
        return final_mask

    min_len = 1
    max_span_len = max(1, int(float(mean_span_length) * 2))
    lengths = torch.arange(min_len, max_span_len + 1, device=device, dtype=torch.float32)
    std = float(mean_span_length) / 3.0
    span_length_probs = torch.exp(-0.5 * ((lengths - float(mean_span_length)) / std) ** 2)
    span_length_probs = span_length_probs / span_length_probs.sum()
    span_lengths_range = lengths.long()

    span_length_indices = torch.multinomial(
        span_length_probs.expand(batch_size, -1),
        num_samples=max_target,
        replacement=True,
    )
    sampled_lengths = span_lengths_range[span_length_indices]
    cumsum_lengths = sampled_lengths.cumsum(dim=1)
    num_spans = (cumsum_lengths < target_counts.unsqueeze(1)).sum(dim=1) + 1

    capacity = (valid_lengths - target_counts + 1).clamp_min(1)
    num_spans = torch.minimum(num_spans, capacity)
    max_spans = int(num_spans.max().item())
    span_slots = torch.arange(max_spans, device=device).view(1, -1)
    span_active = span_slots < num_spans.unsqueeze(1)

    span_lengths = sampled_lengths[:, :max_spans].clone()
    span_lengths = span_lengths * span_active.long()
    span_sums = span_lengths.sum(dim=1)
    overflow = (span_sums - target_counts).clamp_min(0)
    last_span = (num_spans - 1).clamp_min(0)
    span_lengths.scatter_add_(1, last_span[:, None], -overflow[:, None])

    masked_counts = span_lengths.sum(dim=1)
    extra_gaps = (
        valid_lengths - masked_counts - (num_spans - 1).clamp_min(0)
    ).clamp_min(0)

    gap_slots = torch.arange(max_spans + 1, device=device).view(1, -1)
    gap_active = gap_slots <= num_spans.unsqueeze(1)
    gap_weights = torch.rand(batch_size, max_spans + 1, device=device)
    gap_weights = gap_weights.masked_fill(~gap_active, 0.0)
    gap_weights = gap_weights / gap_weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
    gaps = (gap_weights * extra_gaps.unsqueeze(1)).floor().long()
    if max_spans > 1:
        interior_gap = (span_slots[:, 1:] < num_spans.unsqueeze(1)).long()
        gaps[:, 1:max_spans] += interior_gap

    previous_lengths = torch.zeros_like(span_lengths)
    previous_lengths[:, 1:] = span_lengths[:, :-1].cumsum(dim=1)
    starts = gaps[:, :max_spans].cumsum(dim=1) + previous_lengths

    offsets = torch.arange(max_span_len, device=device).view(1, 1, -1)
    span_indices = starts.unsqueeze(-1) + offsets
    token_active = span_active.unsqueeze(-1) & (offsets < span_lengths.unsqueeze(-1))
    batch_indices = (
        torch.arange(batch_size, device=device)
        .view(-1, 1, 1)
        .expand_as(span_indices)
    )
    final_mask[batch_indices[token_active], span_indices[token_active]] = True

    return final_mask & attention_mask


def collate_align_train(
    batch: List[Tuple],
    max_seq_len: int,
):
    vectors, _, map_features, beatmap_ids, targets = zip(*batch)
    labels = _alignment_labels(map_features, beatmap_ids, targets)
    labels["use_contrastive"] = True

    vector_batch = batch_packed_vectors(vectors, max_seq_len)
    return {
        **vector_batch,
        "max_seqlen": torch.tensor(vector_batch["max_seqlen"], dtype=torch.long),
        "labels": labels,
        "batch_size": len(batch),
    }


def collate_align_eval(
    batch: List[Tuple],
    max_seq_len: int,
):
    vectors, _, map_features, beatmap_ids, targets = zip(*batch)
    labels = _alignment_labels(map_features, beatmap_ids, targets)
    labels["use_contrastive"] = False

    max_len = max(min(v.shape[0], max_seq_len) for v in vectors)
    vector_batch = batch_padded_vectors(
        vectors,
        max_seq_len,
        pad_to_len=rounded_pad_length(max_len, max_seq_len, []),
    )
    return {**vector_batch, "labels": labels, "batch_size": len(batch)}


def collate_adapter_train(batch: List[Tuple]):
    embeddings, beatmap_ids, targets = zip(*batch)
    labels = _alignment_labels(None, beatmap_ids, targets)
    labels["use_contrastive"] = True
    return {
        "embeddings": torch.stack(embeddings, dim=0),
        "labels": labels,
        "batch_size": len(batch),
    }


def collate_adapter_eval(batch: List[Tuple]):
    embeddings, beatmap_ids, targets = zip(*batch)
    labels = _alignment_labels(None, beatmap_ids, targets)
    labels["use_contrastive"] = False
    return {
        "embeddings": torch.stack(embeddings, dim=0),
        "labels": labels,
        "batch_size": len(batch),
    }
