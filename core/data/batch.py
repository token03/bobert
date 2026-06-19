import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import torch
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


def rounded_pad_length(
    length: int, max_seq_len: int, buckets: Sequence[int]
) -> int:
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
    vector_dim: int,
) -> Dict[str, torch.Tensor | int]:
    _, lengths, cu_seqlens = _batch_lengths(
        vectors,
        max_seq_len,
        pad_to_len=max_seq_len,
    )
    packed_vectors = torch.zeros(sum(lengths), vector_dim, dtype=torch.float32)
    offset = 0
    for vector, length in zip(vectors, lengths):
        packed_vectors[offset : offset + length] = vector[:length]
        offset += length
    return {
        "packed_vectors": packed_vectors,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max(lengths) if lengths else 0,
    }


def batch_padded_vectors(
    vectors: Sequence[torch.Tensor],
    max_seq_len: int,
    vector_dim: int,
    *,
    pad_to_len: int,
) -> Dict[str, torch.Tensor]:
    effective_max_seq_len, lengths, cu_seqlens = _batch_lengths(
        vectors,
        max_seq_len,
        pad_to_len=pad_to_len,
    )

    max_len = effective_max_seq_len
    padded = torch.zeros(len(vectors), max_len, vector_dim, dtype=torch.float32)
    mask = torch.zeros(len(vectors), max_len, dtype=torch.bool)
    for i, (vector, length) in enumerate(zip(vectors, lengths)):
        padded[i, :length] = vector[:length]
        mask[i, :length] = True
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


def _alignment_labels(
    map_features: Tuple[Dict[str, Any], ...],
    beatmap_ids: Tuple[int, ...],
    targets: Tuple[Dict[str, Any], ...],
):
    id_to_batch = {int(bid): i for i, bid in enumerate(beatmap_ids)}
    positive_weights = torch.zeros(len(targets), len(targets), dtype=torch.float32)
    anchor_weights = torch.tensor(
        [float(target["anchor_weight"]) for target in targets],
        dtype=torch.float32,
    )
    beatmapset_ids = torch.tensor(
        [int(target["beatmapset_id"]) for target in targets], dtype=torch.long
    )
    valid_sets = beatmapset_ids >= 0
    same_set = beatmapset_ids[:, None] == beatmapset_ids[None, :]
    song_keys = [str(target["song_key"]) for target in targets]
    same_song = torch.zeros(len(targets), len(targets), dtype=torch.bool)
    parsed_song_keys = [set(key.split("|")) - {""} for key in song_keys]
    for i, left in enumerate(parsed_song_keys):
        if not left:
            continue
        for j, right in enumerate(parsed_song_keys):
            same_song[i, j] = bool(left & right)

    ignore_contrastive = (
        same_set & valid_sets[:, None] & valid_sets[None, :]
    ) | same_song
    ignore_contrastive.fill_diagonal_(False)

    for i, target in enumerate(targets):
        for bid in target["ignore_ids"]:
            j = id_to_batch.get(int(bid))
            if j is not None and j != i:
                ignore_contrastive[i, j] = True
        for bid, weight in zip(
            target["graph_positive_ids"], target["graph_positive_weights"]
        ):
            j = id_to_batch.get(int(bid))
            if j is not None and j != i:
                existing = float(positive_weights[i, j])
                new_weight = max(float(weight), 0.0)
                positive_weights[i, j] = 1.0 - (1.0 - existing) * (1.0 - new_weight)
                ignore_contrastive[i, j] = True

    return {
        "positive_weights": positive_weights,
        "ignore_contrastive": ignore_contrastive,
        "anchor_weights": anchor_weights,
        "map_features": stack_map_features(list(map_features)),
    }


def collate_pretrain(
    batch: List[Tuple[torch.Tensor, Dict[str, float]]],
    max_seq_len: int,
    vector_dim: int,
    length_buckets: Sequence[int],
):
    vectors, attrs = zip(*batch)
    max_len = max(min(v.shape[0], max_seq_len) for v in vectors)
    vector_batch = batch_padded_vectors(
        vectors,
        max_seq_len,
        vector_dim,
        pad_to_len=rounded_pad_length(max_len, max_seq_len, length_buckets),
    )
    return (
        vector_batch["vectors"],
        vector_batch["attention_mask"],
        stack_dicts(attrs),
        vector_batch["cu_seqlens"],
    )


def collate_align_train(
    batch: List[Tuple],
    max_seq_len: int,
    vector_dim: int,
):
    vectors, _, map_features, beatmap_ids, targets = zip(*batch)
    labels = _alignment_labels(map_features, beatmap_ids, targets)
    labels["use_contrastive"] = True

    vector_batch = batch_packed_vectors(vectors, max_seq_len, vector_dim)
    return {
        **vector_batch,
        "max_seqlen": torch.tensor(vector_batch["max_seqlen"], dtype=torch.long),
        "labels": labels,
        "batch_size": len(batch),
    }


def collate_align_eval(
    batch: List[Tuple],
    max_seq_len: int,
    vector_dim: int,
):
    vectors, _, map_features, beatmap_ids, targets = zip(*batch)
    labels = _alignment_labels(map_features, beatmap_ids, targets)
    labels["use_contrastive"] = False

    max_len = max(min(v.shape[0], max_seq_len) for v in vectors)
    vector_batch = batch_padded_vectors(
        vectors,
        max_seq_len,
        vector_dim,
        pad_to_len=rounded_pad_length(max_len, max_seq_len, []),
    )
    return {**vector_batch, "labels": labels, "batch_size": len(batch)}
