import math
import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Sampler


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


def batch_packed_vectors(
    vectors: Sequence[torch.Tensor], max_seq_len: int
) -> Dict[str, torch.Tensor | int]:
    lengths = [min(int(vector.shape[0]), int(max_seq_len)) for vector in vectors]
    seqlens = torch.tensor(lengths, dtype=torch.int32)
    return {
        "packed_vectors": torch.cat(
            [vector[:length] for vector, length in zip(vectors, lengths)], dim=0
        ),
        "cu_seqlens": torch.nn.functional.pad(
            torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
        ),
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


def collate_pretrain(
    batch: List[torch.Tensor],
    max_seq_len: int,
    masking_ratio: float,
    mean_span_length: float,
    q_buckets: Sequence[int],
):
    lengths = [min(int(vector.shape[0]), int(max_seq_len)) for vector in batch]
    seqlens = torch.tensor(lengths, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    masks = [span_mask(length, masking_ratio, mean_span_length) for length in lengths]
    masked_counts = torch.tensor([int(mask.sum()) for mask in masks], dtype=torch.int32)
    masked_positions = torch.cat(
        [mask.nonzero(as_tuple=False).flatten() for mask in masks], dim=0
    )
    masked_idx = masked_positions + torch.repeat_interleave(
        cu_seqlens[:-1].long(), masked_counts.long()
    )
    split = torch.rand(masked_idx.numel())
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
            [vector[:length] for vector, length in zip(batch, lengths)], dim=0
        ),
        "masked_idx": masked_idx,
        "masked_positions": masked_positions.to(torch.int32),
        "masked_counts": masked_counts,
        "max_seqlen_q": select_q_bucket(int(masked_counts.max()), q_buckets),
        "mask_token_idx": masked_idx[split < 0.8],
        "random_dst_idx": masked_idx[(split >= 0.8) & (split < 0.9)],
        "right_border_zero_idx": right_border_idx[right_split < 0.8],
        "right_border_random_idx": right_border_idx[
            (right_split >= 0.8) & (right_split < 0.9)
        ],
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max(lengths),
    }
