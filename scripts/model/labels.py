import argparse
from functools import partial

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from core.artifacts import (
    Translator,
    embedding_metadata,
    from_bfloat16,
    resolve_artifacts,
    save_labels,
    unit_rows,
)
from core.dataset import FeatureStore, LengthBucketBatchSampler, feature_path
from core.model import EmbeddingTransform
from scripts.common.collections import mode_collections
from scripts.common.paths import COLLECTIONS_DIR, FEATURES_DIR, RUNS_DIR
from scripts.model.embed import ExportDataset, collate_export, load_model
from scripts.sources.collections.tournaments import TOURNAMENT_PATH

WORDS = {
    "std": {
        "speed": r"\bspeed\b",
        "flow aim": r"flow ?aim",
        "aim control": r"aim ?control",
        "aim slop": r"(aim ?)?slop",
        "aim": r"\b(aim|jumps?)\b",
        "tech": r"\btech(nical)?\b",
        "alt": r"\b(alt|alternate|alternating)\b",
        "finger control": r"finger ?control",
        "streams": r"\bstreams?\b",
        "bursts": r"\bbursts?\b",
        "stamina": r"\bstamina\b",
        "reading": r"\bread(ing)?\b",
        "hybrid": r"\bhybrid\b",
        "precision": r"\b(precision|high cs)\b",
        "gimmick": r"\bgimmick\b",
        "consistency": r"\bconsisten(cy|t)\b",
    },
}
QUALIFIERS = {"std": ["consistency"]}
DT_SLOT_WORDS = {"std": {1: "aim", 2: "speed"}}
SLOT_WEIGHT = 0.5
FAST_NAME = r"\b(dt|nc|hddt|dthr|ezdt|doubletime|double ?time|dt\d+|nc\d+)\b"
NORMAL_NAME = r"\b(nm|nomod|no ?mod|hr|hdhr|hd|nm\d+|hr\d+|hd\d+)\b"
RATES = {"fast": 1.5, "slow": 0.75}
DURATION_FIELDS = (
    "log_onset_ioi_ms",
    "log_span_duration_ms",
    "log_spinner_duration_ms",
)
TRANSLATOR_SAMPLE = 30000
TRANSLATOR_RIDGE = 0.1
INTENT_RIDGE = 0.1
WORD_RIDGE = 1.0
WORD_BALANCE = 0.25
MIN_COLLECTIONS = 40
MEMBERS_PER_COLLECTION = 200
CENTER_SAMPLE = 20000
BATCH_TOKENS = 64 * 512
SEED = 42


class RateDataset(ExportDataset):
    def __init__(self, store: FeatureStore, positions, max_seq_len: int, rate: float):
        super().__init__(store, positions, max_seq_len)
        self.rate = rate
        self.columns = [store.field_names.index(name) for name in DURATION_FIELDS]

    def __getitem__(self, idx):
        beatmap_id, rows = super().__getitem__(idx)
        rows = rows.float()
        durations = rows[:, self.columns]
        rows[:, self.columns] = torch.log1p(torch.expm1(durations) / self.rate)
        return beatmap_id, rows


def ridge(inputs: np.ndarray, targets: np.ndarray, strength: float) -> np.ndarray:
    gram = inputs.T @ inputs + strength * np.eye(inputs.shape[1], dtype=inputs.dtype)
    return np.linalg.solve(gram, inputs.T @ targets)


def embed_at_rate(model, transform, store, positions, rate, device):
    lengths = np.minimum(store.lengths[positions], model.max_seq_len)
    loader = DataLoader(
        RateDataset(store, positions, model.max_seq_len, rate),
        batch_sampler=LengthBucketBatchSampler(
            lengths.tolist(),
            batch_size=None,
            max_tokens=BATCH_TOKENS,
            seed=SEED,
            shuffle=False,
        ),
        pin_memory=True,
        collate_fn=partial(collate_export, model=model),
    )
    dtype = next(model.parameters()).dtype
    ids, vectors = [], []
    with torch.inference_mode():
        for beatmap_ids, inputs, cu_seqlens, max_seqlen in loader:
            layers = model.embed_packed(
                inputs.to(device=device, dtype=dtype, non_blocking=True),
                cu_seqlens.to(device, non_blocking=True),
                max_seqlen,
            )
            vectors.append(
                model.transform(
                    layers.permute(1, 0, 2).float().cpu().numpy(), transform
                )
            )
            ids.append(beatmap_ids.numpy())
    return np.concatenate(ids), unit_rows(np.concatenate(vectors))


def read_embeddings(index_path, beatmap_ids: pl.Series) -> dict[int, np.ndarray]:
    frame = (
        pl.scan_parquet(index_path)
        .filter(pl.col("beatmap_id").is_in(beatmap_ids.implode()))
        .select("beatmap_id", "embedding")
        .collect()
    )
    vectors = unit_rows(from_bfloat16(np.stack(frame["embedding"].to_numpy())))
    return dict(zip(frame["beatmap_id"].to_list(), vectors, strict=True))


def fit_translators(artifacts, index_ids: pl.Series) -> dict[str, Translator]:
    device = torch.device("cuda")
    model = load_model(artifacts.model, device)
    transform = EmbeddingTransform(
        np.asarray(embedding_metadata(artifacts.index)["layer_means"], dtype=np.float32)
    )
    store = FeatureStore(feature_path(FEATURES_DIR, artifacts.mode), artifacts.mode)
    stored = pl.Series(store.beatmap_ids)
    candidates = np.flatnonzero(stored.is_in(index_ids.implode()).to_numpy())
    positions = np.sort(
        np.random.default_rng(SEED).choice(candidates, TRANSLATOR_SAMPLE, replace=False)
    )
    normal = read_embeddings(artifacts.index, pl.Series(store.beatmap_ids[positions]))
    translators = {}
    for speed, rate in RATES.items():
        ids, played = embed_at_rate(model, transform, store, positions, rate, device)
        inputs = np.stack([normal[beatmap_id] for beatmap_id in ids])
        train = ids % 10 != 0
        input_mean, played_mean = inputs[train].mean(0), played[train].mean(0)
        weight = ridge(
            inputs[train] - input_mean, played[train] - played_mean, TRANSLATOR_RIDGE
        )
        translators[speed] = Translator(weight, played_mean - input_mean @ weight)
        fit = (translators[speed].apply(inputs[~train]) * played[~train]).sum(1)
        print(
            f"{speed} translator: held-out cos median {np.median(fit):.3f}, "
            f"p5 {np.percentile(fit, 5):.3f} (untranslated "
            f"{np.median((inputs[~train] * played[~train]).sum(1)):.3f})"
        )
    return translators


def read_collections(mode: str, index_ids: pl.Series, words: dict[str, str]):
    names = pl.read_parquet(
        COLLECTIONS_DIR / "vertices.parquet",
        columns=["collection_id", "source", "name"],
    ).with_columns(pl.col("name").str.to_lowercase())
    return (
        mode_collections(mode)
        .join(names, on=["collection_id", "source"])
        .explode("beatmap_id")
        .join(index_ids.to_frame(), on="beatmap_id", how="semi")
        .group_by("collection_id", "source", "name")
        .agg(pl.col("beatmap_id").sample(fraction=1.0, shuffle=True, seed=SEED))
        .with_columns(
            pl.col("beatmap_id").list.head(MEMBERS_PER_COLLECTION),
            pl.col("name").str.contains(FAST_NAME).alias("fast_name"),
            pl.col("name").str.contains(NORMAL_NAME).alias("normal_name"),
            *(pl.col("name").str.contains(p).alias(word) for word, p in words.items()),
        )
        .filter(
            pl.any_horizontal(list(words))
            | (pl.col("fast_name") ^ pl.col("normal_name"))
        )
        .sort("source", "collection_id")
    )


def read_dt_slots(index_ids: pl.Series, slot_words: dict[int, str]) -> pl.DataFrame:
    return (
        pl.read_parquet(
            TOURNAMENT_PATH, columns=["tournament_id", "mod", "map_index", "beatmap_id"]
        )
        .filter(pl.col("mod").str.strip_chars().str.to_uppercase() == "DT")
        .with_columns((pl.col("map_index") + 1).alias("slot"))
        .filter(pl.col("slot").is_in(list(slot_words)))
        .join(index_ids.to_frame(), on="beatmap_id", how="semi")
        .group_by("tournament_id", "slot")
        .agg(pl.col("beatmap_id").unique())
        .with_columns(pl.col("slot").replace_strict(slot_words).alias("word"))
        .sort("tournament_id", "slot")
    )


def group_means(embeddings, groups: pl.Series, translator=None) -> np.ndarray:
    means = []
    for ids in groups:
        vectors = np.stack([embeddings[beatmap_id] for beatmap_id in ids])
        means.append((translator.apply(vectors) if translator else vectors).mean(0))
    return np.stack(means)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cluster labels from collection names"
    )
    parser.add_argument("-v", "--version", default="current")
    parser.add_argument("--mode", choices=sorted(WORDS), default="std")
    args = parser.parse_args()
    artifacts = resolve_artifacts(RUNS_DIR / args.version, args.mode)
    index_ids = (
        pl.scan_parquet(artifacts.index).select("beatmap_id").collect()["beatmap_id"]
    )
    translators = fit_translators(artifacts, index_ids)

    collections = read_collections(args.mode, index_ids, WORDS[args.mode])
    slots = read_dt_slots(index_ids, DT_SLOT_WORDS[args.mode])
    sample = index_ids.sample(min(CENTER_SAMPLE, len(index_ids)), seed=SEED)
    embeddings = read_embeddings(
        artifacts.index,
        pl.concat(
            [sample, collections["beatmap_id"].explode(), slots["beatmap_id"].explode()]
        ),
    )
    center = np.stack([embeddings[beatmap_id] for beatmap_id in sample]).mean(0)
    normal = group_means(embeddings, collections["beatmap_id"])
    fast = group_means(embeddings, collections["beatmap_id"], translators["fast"])

    fast_name = collections["fast_name"].to_numpy()
    named = fast_name ^ collections["normal_name"].to_numpy()
    offsets = unit_rows(normal[named] - center)
    intent_mean = fast_name[named].mean()
    intent = ridge(offsets, fast_name[named] - intent_mean, INTENT_RIDGE)
    unmarked_fast = unit_rows(normal - center) @ intent + intent_mean > 0.5
    is_fast = np.where(named, fast_name, unmarked_fast)
    print(f"DT-intended: {is_fast[~named].sum():,} of {(~named).sum():,} unmarked")

    names = list(WORDS[args.mode])
    members = collections.select(names).to_numpy().astype(np.float32)
    for word, count in zip(names, members.sum(0), strict=True):
        if count < MIN_COLLECTIONS:
            raise ValueError(f"too few collections for {word}: {int(count)}")
        print(f"{word}: {int(count):,} collections")
    print(f"DT tournament slots: {slots.height:,}")
    labelled = members.sum(1) > 0
    played = unit_rows(
        np.vstack(
            [
                np.where(is_fast[:, None], fast, normal)[labelled],
                group_means(embeddings, slots["beatmap_id"], translators["fast"]),
            ]
        )
        - center
    )
    slot_members = slots["word"].to_numpy()[:, None] == np.array(names)
    members = np.vstack([members[labelled], slot_members.astype(np.float32)])
    sample_weight = np.r_[np.ones(labelled.sum()), np.full(slots.height, SLOT_WEIGHT)]
    targets = members / members.sum(0) ** WORD_BALANCE
    targets -= np.average(targets, axis=0, weights=sample_weight)
    scale = np.sqrt(sample_weight)[:, None]
    weights = ridge(played * scale, targets * scale, WORD_RIDGE)
    bias = -np.average(played @ weights, axis=0, weights=sample_weight)
    save_labels(
        artifacts.labels,
        names,
        QUALIFIERS[args.mode],
        weights,
        bias,
        center,
        translators,
        embedding_metadata(artifacts.index),
    )
    print(f"Wrote {len(names)} labels to {artifacts.labels}")


if __name__ == "__main__":
    main()
