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
        self, ids: List[int], weights: List[float], rng: random.Random
    ) -> Optional[int]:
        available = []
        available_weights = []
        if len(weights) != len(ids):
            weights = [1.0] * len(ids)
        for bid, weight in zip(ids, weights):
            bid = int(bid)
            if bid in self.id_to_idx:
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

            p1 = self._choose_id(positive_ids, positive_weights, rng)
            p2 = self._choose_id(cross_ids, cross_weights, rng) or self._choose_id(
                positive_ids, positive_weights, rng
            )
            n1 = self._choose_id(negative_ids, negative_weights, rng)

            group = [anchor_idx]
            group_ids = {anchor_id}
            for chosen in [p1, p2, n1]:
                if chosen is not None and chosen not in group_ids:
                    group.append(self.id_to_idx[chosen])
                    group_ids.add(chosen)

            while len(group) < self.group_size:
                group.append(rng.randrange(len(self.beatmap_ids)))

            batch.extend(group[: self.group_size])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
