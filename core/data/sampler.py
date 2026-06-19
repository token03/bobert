import math
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence

from torch.utils.data import Sampler


def length_bucket(length: int, buckets: Sequence[int]) -> int:
    for bucket in buckets:
        if length <= bucket:
            return int(bucket)
    return int(buckets[-1])


class LengthBucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        buckets: Sequence[int],
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        if not buckets:
            raise ValueError("length buckets must not be empty")
        if sorted(buckets) != list(buckets):
            raise ValueError("length buckets must be sorted in ascending order")

        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.buckets = [int(bucket) for bucket in buckets]
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        counts = {bucket: 0 for bucket in self.buckets}
        for length in self.lengths:
            counts[length_bucket(length, self.buckets)] += 1

        total = 0
        for bucket, count in counts.items():
            limit = self._bucket_batch_size(bucket)
            if self.drop_last:
                total += count // limit
            else:
                total += (count + limit - 1) // limit
        return max(1, total)

    def _bucket_batch_size(self, bucket: int) -> int:
        if self.max_tokens is None:
            return self.batch_size
        size = max(1, min(self.batch_size, self.max_tokens // bucket))
        if size >= 8:
            size = max(8, (size // 8) * 8)
        return size

    def _indices(self) -> Iterator[int]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(None if self.seed is None else self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)
        self.epoch += 1
        for idx in indices:
            yield idx

    def __iter__(self):
        pending: Dict[int, List[int]] = {bucket: [] for bucket in self.buckets}

        for idx in self._indices():
            bucket = length_bucket(self.lengths[idx], self.buckets)
            batch = pending[bucket]
            batch.append(idx)
            if len(batch) == self._bucket_batch_size(bucket):
                yield batch.copy()
                batch.clear()

        if not self.drop_last:
            for bucket in self.buckets:
                batch = pending[bucket]
                if batch:
                    yield batch.copy()


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
            positive_ids = self._sample_positives(
                mining, rng, self.group_size - 1
            )
            if not positive_ids:
                continue

            batch.extend([anchor_idx, *(self.id_to_idx[bid] for bid in positive_ids)])
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch
