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

    def _choose_ranked_id(
        self,
        ids: List[int],
        exclude: set[int],
        offset: int,
        avoid_beatmapset_id: Optional[int] = None,
    ) -> Optional[int]:
        available = []
        for bid in ids:
            bid = int(bid)
            if bid not in self.id_to_idx or bid in exclude:
                continue
            if avoid_beatmapset_id is not None:
                target = self.mining_lookup.get(bid, {})
                if int(target.get("beatmapset_id", -1)) == avoid_beatmapset_id:
                    continue
            available.append(bid)
        if not available:
            return None
        return available[offset % len(available)]

    def _add_ranked(
        self,
        group: List[int],
        group_ids: set[int],
        ids: List[int],
        offset: int,
        avoid_beatmapset_id: Optional[int] = None,
    ) -> bool:
        bid = self._choose_ranked_id(ids, group_ids, offset, avoid_beatmapset_id)
        if bid is None:
            return False
        group.append(self.id_to_idx[bid])
        group_ids.add(bid)
        return True

    def _add_random(
        self,
        group: List[int],
        group_ids: set[int],
        rng: random.Random,
        avoid_beatmapset_id: Optional[int] = None,
    ) -> bool:
        for _ in range(max(1, len(self.beatmap_ids) * 2)):
            random_idx = rng.randrange(len(self.beatmap_ids))
            random_id = self.beatmap_ids[random_idx]
            if len(group_ids) < len(self.beatmap_ids) and random_id in group_ids:
                continue
            if avoid_beatmapset_id is not None:
                target = self.mining_lookup.get(random_id, {})
                if int(target.get("beatmapset_id", -1)) == avoid_beatmapset_id:
                    continue
            group.append(random_idx)
            group_ids.add(random_id)
            return True
        return False

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

            graph_ids = mining.get("graph_positive_ids", [])
            song_ids = mining.get("song_positive_ids", [])
            creator_ids = mining.get("creator_positive_ids", [])
            cross_ids = mining.get("cross_status_positive_ids", [])
            anchor_set_id = int(mining.get("beatmapset_id", -1))
            offset = max(0, self.epoch - 1)

            group = [anchor_idx]
            group_ids = {anchor_id}

            if len(group) < self.group_size:
                added = self._add_ranked(
                    group,
                    group_ids,
                    graph_ids,
                    offset,
                    avoid_beatmapset_id=anchor_set_id if anchor_set_id >= 0 else None,
                )
                if not added:
                    added = self._add_ranked(
                        group,
                        group_ids,
                        cross_ids,
                        offset,
                        avoid_beatmapset_id=anchor_set_id if anchor_set_id >= 0 else None,
                    )
                if not added:
                    added = self._add_ranked(
                        group,
                        group_ids,
                        creator_ids,
                        offset,
                        avoid_beatmapset_id=anchor_set_id if anchor_set_id >= 0 else None,
                    )
                if not added:
                    continue

            if len(group) < self.group_size:
                added = self._add_ranked(group, group_ids, song_ids, offset)
                if not added:
                    added = self._add_ranked(group, group_ids, creator_ids, offset)
                if not added:
                    self._add_ranked(
                        group,
                        group_ids,
                        graph_ids,
                        offset + 1,
                        avoid_beatmapset_id=anchor_set_id if anchor_set_id >= 0 else None,
                    )

            if len(group) < self.group_size:
                added = self._add_ranked(
                    group,
                    group_ids,
                    cross_ids,
                    offset,
                    avoid_beatmapset_id=anchor_set_id if anchor_set_id >= 0 else None,
                )
                if not added:
                    added = self._add_ranked(
                        group,
                        group_ids,
                        graph_ids,
                        offset + 1,
                        avoid_beatmapset_id=anchor_set_id if anchor_set_id >= 0 else None,
                    )
                if not added:
                    added = self._add_ranked(
                        group,
                        group_ids,
                        creator_ids,
                        offset + 1,
                        avoid_beatmapset_id=anchor_set_id if anchor_set_id >= 0 else None,
                    )
                if not added:
                    self._add_ranked(
                        group,
                        group_ids,
                        graph_ids,
                        offset + 2,
                        avoid_beatmapset_id=anchor_set_id if anchor_set_id >= 0 else None,
                    )

            while len(group) < self.group_size:
                if not self._add_random(
                    group,
                    group_ids,
                    rng,
                    avoid_beatmapset_id=(
                        None
                        if len(group) == 2 or anchor_set_id < 0
                        else anchor_set_id
                    ),
                ):
                    break

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
