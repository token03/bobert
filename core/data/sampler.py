import math
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Sized

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
        sampler: Optional[Iterable[int]] = None,
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
        self.sampler = sampler
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        counts = {bucket: 0 for bucket in self.buckets}
        for length in self.lengths:
            counts[length_bucket(length, self.buckets)] += 1

        sample_count = len(self.lengths)
        if isinstance(self.sampler, Sized):
            sample_count = len(self.sampler)
        scale = sample_count / max(1, len(self.lengths))

        total = 0
        for bucket, count in counts.items():
            count = int(round(count * scale))
            limit = self._bucket_batch_size(bucket)
            if self.drop_last:
                total += count // limit
            else:
                total += (count + limit - 1) // limit
        return max(1, total)

    def _bucket_batch_size(self, bucket: int) -> int:
        if self.max_tokens is None:
            return self.batch_size
        return max(1, min(self.batch_size, self.max_tokens // bucket))

    def _indices(self) -> Iterator[int]:
        if self.sampler is not None:
            for idx in self.sampler:
                yield int(idx)
            return

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
        group_size: int = 4,
        seed: int = 42,
        anchor_indices: Optional[Sequence[int]] = None,
        epoch_size: Optional[int] = None,
        lengths: Optional[Sequence[int]] = None,
        buckets: Optional[Sequence[int]] = None,
        max_tokens: Optional[int] = None,
    ):
        if batch_size % group_size != 0:
            raise ValueError("alignment batch_size must be divisible by group_size")
        if epoch_size is not None and epoch_size <= 0:
            raise ValueError("alignment epoch_size must be positive when provided")

        self.beatmap_ids = [int(x) for x in beatmap_ids]
        self.mining_lookup = mining_lookup
        self.batch_size = batch_size
        self.group_size = group_size
        self.seed = seed
        self.id_to_idx = {bid: i for i, bid in enumerate(self.beatmap_ids)}
        self.anchor_indices = (
            [int(idx) for idx in anchor_indices]
            if anchor_indices is not None
            else list(range(len(self.beatmap_ids)))
        )
        self.epoch_size = int(epoch_size) if epoch_size is not None else None
        self.groups_per_batch = batch_size // group_size
        self.epoch = 0
        self.lengths = (
            [int(length) for length in lengths] if lengths is not None else None
        )
        self.buckets = [int(bucket) for bucket in buckets] if buckets else None
        self.max_tokens = int(max_tokens) if max_tokens else None

    def __len__(self) -> int:
        anchor_count = self._anchor_count()
        if self.lengths is None or self.buckets is None:
            return max(1, math.ceil(anchor_count / self.groups_per_batch))

        counts = {bucket: 0 for bucket in self.buckets}
        for idx in self.anchor_indices:
            counts[length_bucket(self.lengths[idx], self.buckets)] += 1
        if anchor_count < len(self.anchor_indices):
            scale = anchor_count / max(1, len(self.anchor_indices))
            counts = {
                bucket: int(round(count * scale))
                for bucket, count in counts.items()
            }

        total = 0
        for bucket, count in counts.items():
            groups_per_bucket_batch = max(
                1, self._bucket_batch_size(bucket) // self.group_size
            )
            total += math.ceil(count / groups_per_bucket_batch)
        return max(1, total)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _anchor_count(self) -> int:
        if self.epoch_size is None:
            return len(self.anchor_indices)
        return min(self.epoch_size, len(self.anchor_indices))

    def _choose_id(
        self,
        ids: List[int],
        weights: List[float],
        rng: random.Random,
        exclude: Optional[set[int]] = None,
    ) -> Optional[int]:
        available = []
        available_weights = []
        exclude = exclude or set()
        if len(weights) != len(ids):
            weights = [1.0] * len(ids)
        for bid, weight in zip(ids, weights):
            bid = int(bid)
            if bid in self.id_to_idx and bid not in exclude:
                available.append(bid)
                available_weights.append(max(float(weight), 0.0))
        if not available:
            return None
        if sum(available_weights) <= 0.0:
            return rng.choice(available)
        return rng.choices(available, weights=available_weights, k=1)[0]

    def _bucket_batch_size(self, bucket: int) -> int:
        if self.max_tokens is None:
            return self.batch_size

        groups = max(
            1,
            min(self.groups_per_batch, self.max_tokens // (bucket * self.group_size)),
        )
        return groups * self.group_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        anchor_indices = self.anchor_indices.copy()
        rng.shuffle(anchor_indices)
        anchor_indices = anchor_indices[: self._anchor_count()]

        batches: Dict[int, List[int]] = (
            {bucket: [] for bucket in self.buckets} if self.buckets else {}
        )
        batch: List[int] = []
        for anchor_idx in anchor_indices:
            anchor_id = self.beatmap_ids[anchor_idx]
            mining = self.mining_lookup.get(anchor_id, {})

            positive_ids = mining.get(
                "target_positive_ids", mining.get("positive_ids", [])
            )
            positive_weights = mining.get(
                "target_positive_weights", mining.get("positive_weights", [])
            )
            cross_ids = mining.get(
                "target_cross_status_positive_ids",
                mining.get("cross_status_positive_ids", []),
            )
            cross_weights = mining.get(
                "target_cross_status_positive_weights",
                mining.get("cross_status_positive_weights", []),
            )
            negative_ids = mining.get("hard_negative_ids", [])
            negative_weights = mining.get("hard_negative_weights", [])

            group = [anchor_idx]
            group_ids = {anchor_id}

            p1 = self._choose_id(positive_ids, positive_weights, rng, group_ids)
            if p1 is None:
                p1 = self._choose_id(cross_ids, cross_weights, rng, group_ids)

            if p1 is not None:
                group.append(self.id_to_idx[p1])
                group_ids.add(p1)

            negative_slots = max(0, self.group_size - 2)
            for _ in range(negative_slots):
                neg = self._choose_id(negative_ids, negative_weights, rng, group_ids)
                if neg is None:
                    break
                group.append(self.id_to_idx[neg])
                group_ids.add(neg)

            while len(group) < self.group_size:
                random_idx = rng.randrange(len(self.beatmap_ids))
                random_id = self.beatmap_ids[random_idx]
                if len(group_ids) < len(self.beatmap_ids) and random_id in group_ids:
                    continue
                group.append(random_idx)
                group_ids.add(random_id)

            group = group[: self.group_size]
            if self.lengths is not None and self.buckets is not None:
                group_len = max(self.lengths[idx] for idx in group)
                bucket = length_bucket(group_len, self.buckets)
                bucket_batch = batches[bucket]
                bucket_batch.extend(group)
                if len(bucket_batch) == self._bucket_batch_size(bucket):
                    yield bucket_batch.copy()
                    bucket_batch.clear()
            else:
                batch.extend(group)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if self.lengths is not None and self.buckets is not None:
            for bucket in self.buckets:
                bucket_batch = batches[bucket]
                if bucket_batch:
                    yield bucket_batch.copy()
        elif batch:
            yield batch
