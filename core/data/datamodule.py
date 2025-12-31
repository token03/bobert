import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Dict, List, Optional, Any, Tuple, Union

from .loader import load_hitobjects, load_metadata_stub, load_tags_stub, setup_dataset
from .transforms import (
    BeatmapAugmenter,
    BeatmapNormalizer,
    BeatmapTransform,
    create_normalizer_from_data,
)
from .vocab import TagTokenizer

def _pad_batch(vectors: List[torch.Tensor], max_seq_len: int, vector_dim: int):
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

def _stack_dicts(dict_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not dict_list:
        return {}
    return {
        k: torch.tensor([d[k] for d in dict_list], dtype=torch.float32)
        for k in dict_list[0]
    }

def _random_split_aligned(
    data_sources: Dict[str, Union[List, Dict]], val_split: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    primary_key = next(iter(data_sources))
    total_len = len(data_sources[primary_key])
    
    val_size = int(total_len * val_split)
    train_size = total_len - val_size
    
    indices = torch.randperm(total_len).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    def extract(source, idx_list):
        if isinstance(source, dict):
            return {k: [v[i] for i in idx_list] for k, v in source.items()}
        return [source[i] for i in idx_list]

    train_out = {k: extract(v, train_idx) for k, v in data_sources.items()}
    val_out = {k: extract(v, val_idx) for k, v in data_sources.items()}
    return train_out, val_out

def pretrain_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, float]]],
    max_seq_len: int,
    vector_dim: int,
):
    vectors, attrs = zip(*batch)
    padded, mask, cu_seqlens = _pad_batch(vectors, max_seq_len, vector_dim)
    return padded, mask, _stack_dicts(attrs), cu_seqlens

def align_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict, torch.Tensor, Dict]],
    max_seq_len: int,
    vector_dim: int,
    max_tags: int = 50,
):
    vectors, metadata, tags, attrs = zip(*batch)
    padded_vec, mask, cu_seqlens = _pad_batch(vectors, max_seq_len, vector_dim)

    tag_lens = [min(t.shape[0], max_tags) for t in tags]
    padded_tags = torch.zeros(len(batch), max(tag_lens) if tag_lens else 1, dtype=torch.long)
    for i, (t, l) in enumerate(zip(tags, tag_lens)):
        if l > 0: padded_tags[i, :l] = t[:l]

    return (
        padded_vec, mask, cu_seqlens,
        None, None,  
        padded_tags, _stack_dicts(attrs),
    )

class BeatmapDataset(Dataset):
    def __init__(
        self,
        beatmap_data: List[torch.Tensor],
        transform: BeatmapTransform,
        difficulty_attributes: Optional[Dict[str, list]] = None,
        beatmap_ids: Optional[List[int]] = None,
        metadata: Optional[Dict[int, Dict]] = None,
        tags: Optional[Dict[int, torch.Tensor]] = None,
    ):
        self.beatmap_data = beatmap_data
        self.transform = transform
        self.diff_attrs = difficulty_attributes
        self.beatmap_ids = beatmap_ids
        self.metadata = metadata or {}
        self.tags = tags or {}
        self.has_meta = beatmap_ids is not None

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def __getitem__(self, idx: int):
        vec = self.transform(self.beatmap_data[idx])
        attrs = {k: v[idx] for k, v in self.diff_attrs.items()} if self.diff_attrs else {}

        if not self.has_meta:
            return vec, attrs

        bid = self.beatmap_ids[idx]
        tags = self.tags.get(bid, torch.tensor([0], dtype=torch.long))
        return vec, self.metadata.get(bid, {}), tags, attrs

class BeatmapDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any], section: str, db_path: Optional[str] = None):
        super().__init__()
        self.config = config
        self.section = section
        self.db_path = db_path or config[section].get("db_path")
        self.batch_size = config[section]["batch_size"]
        self.vector_dim: Optional[int] = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        setup_dataset(self.db_path, self.config.get(self.section, {}).get("colab_url"))

    def _setup_common(self) -> Tuple[List, Dict, List]:
        return load_hitobjects(
            self.db_path,
            max_seq_len=self.config["data"]["max_seq_len"],
            raw_beatmap_path=self.config[self.section].get("raw_beatmap_path", "./data/osu"),
        )

    def _create_datasets(self, train_data, val_data) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset: return

        all_data, diff_attrs, all_ids = self._setup_common()
        
        sources = {"data": all_data, "attrs": diff_attrs}
        if all_ids is not None:
            sources["ids"] = all_ids.tolist()

        train_s, val_s = _random_split_aligned(sources, self.config["data"]["val_split"])

        train_attrs_np = {k: np.array(v) for k, v in train_s["attrs"].items()}
        self.normalizer = create_normalizer_from_data(train_s["data"], train_attrs_np)
        
        aug = BeatmapAugmenter()
        self.t_train = BeatmapTransform(self.normalizer, aug, augment=True)
        self.t_val = BeatmapTransform(self.normalizer, aug, augment=False)
        self.vector_dim = train_s["data"][0].shape[1]

        self.train_dataset, self.val_dataset = self._create_datasets(train_s, val_s)
        
        if hasattr(self, "_setup_sampler"):
            self._setup_sampler(train_s["attrs"])

        print(f"Data split: {len(self.train_dataset)} training, {len(self.val_dataset)} validation")

    def _get_dataloader(self, dataset, shuffle, collate_fn, sampler=None):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and not sampler,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.config["data"].get("num_workers", 0),
            pin_memory=True,
        )

class PretrainDataModule(BeatmapDataModule):
    def __init__(self, config, db_path=None, sampler_fn=None):
        super().__init__(config, "pretraining", db_path)
        self.sampler_fn = sampler_fn
        self._sampler = None

    def _setup_sampler(self, train_attrs):
        if self.sampler_fn:
            self._sampler = self.sampler_fn(np.array(train_attrs["stars"]))

    def _create_datasets(self, train_s, val_s):
        return (
            BeatmapDataset(train_s["data"], self.t_train, train_s["attrs"]),
            BeatmapDataset(val_s["data"], self.t_val, val_s["attrs"]),
        )

    def train_dataloader(self):
        collate = lambda b: pretrain_collate_fn(b, self.config["data"]["max_seq_len"], self.vector_dim)
        return self._get_dataloader(self.train_dataset, True, collate, self._sampler)

    def val_dataloader(self):
        collate = lambda b: pretrain_collate_fn(b, self.config["data"]["max_seq_len"], self.vector_dim)
        return self._get_dataloader(self.val_dataset, False, collate)

class AlignDataModule(BeatmapDataModule):
    def __init__(self, config, tag_tokenizer, db_path=None):
        super().__init__(config, "alignment", db_path)
        self.tag_tokenizer = tag_tokenizer
        self.meta_store = {}
        self.tag_store = {}

    def setup(self, stage=None):
        if self.train_dataset: return
        
        all_data, diff_attrs, ids_array = self._setup_common()
        all_ids = ids_array.tolist()
        
        self.meta_store = load_metadata_stub(all_ids)
        tags_raw = load_tags_stub(all_ids)
        self.tag_store = {k: self.tag_tokenizer.encode(v) for k, v in tags_raw.items()}

        sources = {"data": all_data, "attrs": diff_attrs, "ids": all_ids}
        train_s, val_s = _random_split_aligned(sources, self.config["data"]["val_split"])
        
        train_attrs_np = {k: np.array(v) for k, v in train_s["attrs"].items()}
        self.normalizer = create_normalizer_from_data(train_s["data"], train_attrs_np)
        
        aug = BeatmapAugmenter()
        self.t_train = BeatmapTransform(self.normalizer, aug, augment=True)
        self.t_val = BeatmapTransform(self.normalizer, aug, augment=False)
        self.vector_dim = train_s["data"][0].shape[1]

        self.train_dataset = BeatmapDataset(
            train_s["data"], self.t_train, train_s["attrs"], 
            train_s["ids"], self.meta_store, self.tag_store
        )
        self.val_dataset = BeatmapDataset(
            val_s["data"], self.t_val, val_s["attrs"], 
            val_s["ids"], self.meta_store, self.tag_store
        )
        print(f"Data split: {len(self.train_dataset)} training, {len(self.val_dataset)} validation")

    def train_dataloader(self):
        collate = lambda b: align_collate_fn(
            b, self.config["data"]["max_seq_len"], self.vector_dim, self.config["alignment"].get("max_tags", 50)
        )
        return self._get_dataloader(self.train_dataset, True, collate)

    def val_dataloader(self):
        collate = lambda b: align_collate_fn(
            b, self.config["data"]["max_seq_len"], self.vector_dim, self.config["alignment"].get("max_tags", 50)
        )
        return self._get_dataloader(self.val_dataset, False, collate)