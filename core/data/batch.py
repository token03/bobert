import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

from .schema import MAP_FEATURE_ATTRIBUTES


def masked_query_buckets(
    length_buckets: Sequence[int], max_seq_len: int, masking_ratio: float
) -> Tuple[int, ...]:
    lengths = {*map(int, length_buckets), int(max_seq_len)}
    return tuple(
        sorted(
            {
                max(32, ((round(length * masking_ratio) + 31) // 32) * 32)
                for length in lengths
                if length <= max_seq_len
            }
        )
    )


def select_q_bucket(max_masked_count: int, buckets: Sequence[int]) -> int:
    for bound in buckets:
        if max_masked_count <= bound:
            return bound
    raise ValueError(f"Masked sequence too long: {max_masked_count}")


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


def span_mask(length: int, ratio: float, mean_span_length: float) -> torch.Tensor:
    mask = torch.zeros(int(length), dtype=torch.bool)
    target = round(int(length) * float(ratio))
    if target <= 0:
        return mask

    max_span_len = max(1, int(mean_span_length * 2))
    span_lengths = list(range(1, max_span_len + 1))
    std = float(mean_span_length) / 3.0
    weights = [
        math.exp(-0.5 * ((span_len - mean_span_length) / std) ** 2)
        for span_len in span_lengths
    ]
    total_weight = sum(weights)
    weights = [weight / total_weight for weight in weights]

    spans = []
    masked = 0
    while masked < target and len(spans) < max(1, length - target + 1):
        span_len = random.choices(span_lengths, weights=weights, k=1)[0]
        span_len = min(span_len, target - masked)
        if span_len <= 0:
            break
        spans.append(span_len)
        masked += span_len

    if not spans:
        return mask

    interior_gaps = max(0, len(spans) - 1)
    extra_gaps = max(0, int(length) - masked - interior_gaps)
    gap_weights = [random.random() for _ in range(len(spans) + 1)]
    gap_weight_sum = sum(gap_weights) or 1.0
    gaps = [int((weight / gap_weight_sum) * extra_gaps) for weight in gap_weights]
    gaps[-1] += extra_gaps - sum(gaps)

    pos = gaps[0]
    for idx, span_len in enumerate(spans):
        mask[pos : pos + span_len] = True
        pos += span_len
        if idx + 1 < len(spans):
            pos += 1 + gaps[idx + 1]
    return mask


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
    batch: List[torch.Tensor],
    max_seq_len: int,
    masking_ratio: float,
    mean_span_length: float,
    q_buckets: Sequence[int],
):
    vectors = batch
    lengths = [min(int(vector.shape[0]), int(max_seq_len)) for vector in vectors]
    seqlens = torch.tensor(lengths, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    masks = [span_mask(length, masking_ratio, mean_span_length) for length in lengths]
    masked_counts = torch.tensor([int(mask.sum()) for mask in masks], dtype=torch.int32)
    max_seqlen_q = select_q_bucket(int(masked_counts.max()), q_buckets)
    masked_positions = torch.cat(
        [mask.nonzero(as_tuple=False).flatten() for mask in masks], dim=0
    )
    masked_idx = masked_positions + torch.repeat_interleave(
        cu_seqlens[:-1].long(), masked_counts.long()
    )
    split = torch.rand(masked_idx.numel())
    mask_token_idx = masked_idx[split < 0.8]
    random_dst_idx = masked_idx[(split >= 0.8) & (split < 0.9)]
    unchanged_idx = masked_idx[split >= 0.9]
    right_border_idx = torch.cat(
        [
            (mask[:-1] & ~mask[1:]).nonzero(as_tuple=False).flatten()
            + cu_seqlens[i].long()
            + 1
            for i, mask in enumerate(masks)
        ]
    )
    right_split = torch.rand(right_border_idx.numel())
    return {
        "packed_vectors": torch.cat(
            [vector[:length] for vector, length in zip(vectors, lengths)], dim=0
        ),
        "masked_idx": masked_idx,
        "masked_positions": masked_positions.to(torch.int32),
        "masked_counts": masked_counts,
        "max_seqlen_q": max_seqlen_q,
        "mask_token_idx": mask_token_idx,
        "random_dst_idx": random_dst_idx,
        "unchanged_idx": unchanged_idx,
        "right_border_zero_idx": right_border_idx[right_split < 0.8],
        "right_border_random_idx": right_border_idx[
            (right_split >= 0.8) & (right_split < 0.9)
        ],
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max(lengths),
        "batch_size": len(batch),
    }


def collate_align_train(
    batch: List[Tuple],
    max_seq_len: int,
):
    vectors, map_features, beatmap_ids, targets = zip(*batch)
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
    vectors, map_features, beatmap_ids, targets = zip(*batch)
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
