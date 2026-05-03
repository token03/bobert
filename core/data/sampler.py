import random
from typing import Any, Dict, List, Optional

from torch.utils.data import Sampler


class AlignmentBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        beatmap_ids: List[int],
        mining_lookup: Dict[int, Dict[str, Any]],
        batch_size: int,
        group_size: int = 4,
        seed: int = 42,
    ):
        if batch_size % group_size != 0:
            raise ValueError("alignment batch_size must be divisible by group_size")

        self.beatmap_ids = [int(x) for x in beatmap_ids]
        self.mining_lookup = mining_lookup
        self.batch_size = batch_size
        self.group_size = group_size
        self.seed = seed
        self.id_to_idx = {bid: i for i, bid in enumerate(self.beatmap_ids)}
        self.groups_per_batch = batch_size // group_size

    def __len__(self) -> int:
        return max(1, len(self.beatmap_ids) // self.batch_size)

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

    def __iter__(self):
        rng = random.Random(self.seed)
        anchor_indices = list(range(len(self.beatmap_ids)))
        rng.shuffle(anchor_indices)

        batch: List[int] = []
        for anchor_idx in anchor_indices:
            anchor_id = self.beatmap_ids[anchor_idx]
            mining = self.mining_lookup.get(anchor_id, {})

            positive_ids = mining.get("positive_ids", [])
            positive_weights = mining.get("positive_weights", [])
            cross_ids = mining.get("cross_status_positive_ids", [])
            cross_weights = mining.get("cross_status_positive_weights", [])
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

            batch.extend(group[: self.group_size])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
