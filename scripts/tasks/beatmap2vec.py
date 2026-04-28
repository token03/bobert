import sys
import os
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class StatusClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(StatusClassifier, self).__init__()
        self.grl = GradientReversal(lambda_=ADV_LAMBDA)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(self.grl(x))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
COLLECTIONS_DATA_PATH = COLLECTIONS_DIR / "collection_beatmaps.parquet"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
COLLECTION_FILTER_PATH = COLLECTIONS_DIR / "collection_filter.json"

VERSION = "v1"
MIN_MAPS_IN_COLLECTION = 10
MAX_MAPS_IN_COLLECTION = 3000
MIN_COLLECTIONS_PER_MAP = 2
JACCARD_THRESHOLD = 0.75
SONG_DAMPENING_POWER = 0.95

EMBEDDING_DIM = 64
# Optimizations:
SAMPLE_RATE = 20  # Samples per item in collection (Dynamic sampling)
MAX_SAMPLES_PER_COLLECTION = 4000  # Cap to prevent OOM on huge cols
BATCH_SIZE_COLS = 64  # Batch size in terms of COLLECTIONS (not pairs)
LR = 0.001
EPOCHS = 50
ADV_LAMBDA = 0.1
NEG_SAMPLES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deduplicate_collections(df, threshold, probe_items=32):
    df["collection_key"] = list(zip(df["collection_id"], df["source"]))
    col_groups = df.groupby("collection_key")["beatmap_id"].apply(set).to_dict()

    content_hashes = {}
    for ckey, bms in col_groups.items():
        sig = tuple(sorted(bms))
        if sig not in content_hashes or ckey[0] < content_hashes[sig][0]:
            content_hashes[sig] = ckey

    unique_content_ckeys = set(content_hashes.values())
    sorted_ckeys = sorted(
        unique_content_ckeys, key=lambda x: len(col_groups[x]), reverse=True
    )

    kept_ckeys = []
    kept_sets = {}
    postings = defaultdict(list)

    for ckey in tqdm(sorted_ckeys, desc="Deduplicating"):
        A = col_groups[ckey]
        len_a = len(A)

        if len_a < 5:
            continue

        items = list(A)
        if len(items) > 4 * probe_items:
            step = max(1, len(items) // (4 * probe_items))
            items = items[::step]

        items.sort(key=lambda x: len(postings.get(x, [])))
        probe = items[:probe_items]

        overlap_counts = defaultdict(int)
        for bm in probe:
            for kc in postings.get(bm, []):
                overlap_counts[kc] += 1

        is_duplicate = False
        if overlap_counts:
            candidates = sorted(
                overlap_counts.items(), key=lambda kv: kv[1], reverse=True
            )[:5]
            for kept_ckey, _ in candidates:
                B = kept_sets[kept_ckey]
                len_b = len(B)

                if len_a < threshold * len_b or len_b < threshold * len_a:
                    continue

                inter = len(A & B)
                union = len_a + len_b - inter
                if inter / union >= threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            kept_ckeys.append(ckey)
            kept_sets[ckey] = A
            for bm in A:
                postings[bm].append(ckey)

    return df[df["collection_key"].isin(kept_ckeys)].copy()


def load_and_process_data(source_filter=None):
    df = pd.read_parquet(COLLECTIONS_DATA_PATH)

    if source_filter:
        df = df[df["source"] == source_filter].copy()

    if os.path.exists(COLLECTION_FILTER_PATH):
        with open(COLLECTION_FILTER_PATH, "r") as f:
            filter_data = json.load(f)
        if "collections" in filter_data:
            bad_ids = []
            for src, ids in filter_data["collections"].items():
                bad_ids.extend(ids)
            df = df[~df["collection_id"].isin(bad_ids)]

    beatmaps_df = pd.read_parquet(
        BEATMAPS_PATH, columns=["id", "beatmapset_id", "mode", "status"]
    )
    df = df.merge(
        beatmaps_df.rename(columns={"id": "beatmap_id"}), on="beatmap_id", how="left"
    )

    df = df[df["mode"] == "osu"].copy()

    col_counts = df.groupby("collection_id")["beatmap_id"].count()
    valid_cols = col_counts[
        (col_counts >= MIN_MAPS_IN_COLLECTION) & (col_counts <= MAX_MAPS_IN_COLLECTION)
    ].index
    df = df[df["collection_id"].isin(valid_cols)].copy()

    df = deduplicate_collections(df, JACCARD_THRESHOLD)

    bm_counts = df["beatmap_id"].value_counts()
    valid_maps = bm_counts[(bm_counts >= MIN_COLLECTIONS_PER_MAP)].index
    df = df[df["beatmap_id"].isin(valid_maps)].copy()

    if df["beatmapset_id"].isna().sum() > 0:
        df = df.dropna(subset=["beatmapset_id"]).copy()

    df["song_occurrence"] = df.groupby(["collection_id", "beatmapset_id"])[
        "beatmap_id"
    ].transform("count")

    df["weight"] = 1.0 / (df["song_occurrence"] ** SONG_DAMPENING_POWER)

    unique_collections = sorted(df["collection_key"].unique())
    unique_beatmaps = sorted(df["beatmap_id"].unique())
    bm_to_idx = {bid: i for i, bid in enumerate(unique_beatmaps)}

    status_map = df.groupby("beatmap_id")["status"].first()
    status_labels = torch.zeros(
        len(unique_beatmaps), dtype=torch.float32, device=DEVICE
    )
    for bid, idx in bm_to_idx.items():
        if bid in status_map:
            status = status_map[bid]
            status_labels[idx] = 1.0 if status in ["1", "2", "3", "4"] else 0.0

    # Create mapping from beatmap_id to beatmapset_id
    bm_to_set_id = pd.Series(df.beatmapset_id.values, index=df.beatmap_id).to_dict()

    return df, unique_beatmaps, bm_to_idx, status_labels, bm_to_set_id


class Beatmap2Vec(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(Beatmap2Vec, self).__init__()
        self.target_embedding = nn.Embedding(num_items, embedding_dim)
        self.context_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.target_embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)

    def forward(self, target, context, negatives, weights=None):
        t_emb = self.target_embedding(target)  # B x D
        c_emb = self.context_embedding(context)  # B x D
        n_emb = self.context_embedding(negatives)  # B x Neg x D

        pos_score = torch.sum(t_emb * c_emb, dim=1)  # B
        pos_loss = -F.logsigmoid(pos_score)

        neg_score = torch.bmm(n_emb, t_emb.unsqueeze(2)).squeeze(2)  # B x Neg
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)

        total_loss = pos_loss + neg_loss

        if weights is not None:
            total_loss = total_loss * weights

        return total_loss.mean()

    def get_embeddings(self):
        return self.target_embedding.weight.data


class BeatmapCollectionDataset(Dataset):
    def __init__(self, df, bm_to_idx, bm_to_set_id, sample_rate=10, max_samples=2000):
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        # Store (beatmap_id, song_occurrence) tuples
        self.collections = (
            df.groupby("collection_key")[["beatmap_id", "song_occurrence"]]
            .apply(lambda x: list(zip(x["beatmap_id"], x["song_occurrence"])))
            .tolist()
        )
        self.bm_to_idx = bm_to_idx
        self.bm_to_set_id = bm_to_set_id
        self.num_items = len(bm_to_idx)

    def __len__(self):
        return len(self.collections)

    def __getitem__(self, idx):
        items = self.collections[idx]
        if len(items) < 2:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        # Filter items and map to indices, keeping original ID and occurrence count
        # mapped_item: (index, beatmap_id, song_occurrence)
        mapped_items = [
            (self.bm_to_idx[bid], bid, occ)
            for bid, occ in items
            if bid in self.bm_to_idx
        ]

        if len(mapped_items) < 2:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        # Dynamic sampling: sample proportional to collection size
        # This fixes the issue where large collections were severely undersampled
        num_samples = min(int(len(mapped_items) * self.sample_rate), self.max_samples)
        num_samples = max(num_samples, 20)

        targets = []
        contexts = []
        weights = []

        # Sampling with replacement is efficient and effective for embeddings
        for _ in range(num_samples):
            t_entry = random.choice(mapped_items)
            c_entry = random.choice(mapped_items)

            t_idx, t_bid, t_occ = t_entry
            c_idx, c_bid, c_occ = c_entry

            if t_idx == c_idx and len(mapped_items) > 1:
                c_entry = random.choice(mapped_items)
                c_idx, c_bid, c_occ = c_entry

            targets.append(t_idx)
            contexts.append(c_idx)

            # Downweight if they are from the same beatmapset
            w = 1.0
            t_set = self.bm_to_set_id.get(t_bid)
            c_set = self.bm_to_set_id.get(c_bid)

            if t_set is not None and c_set is not None and t_set == c_set:
                # Use the occurrence count from the collection (should be same for both if in same set)
                # We use max just in case of data inconsistency, though logicaly they are same set in same col.
                cnt = max(t_occ, c_occ)
                if cnt > 1:
                    w = 1.0 / (cnt**SONG_DAMPENING_POWER)

            weights.append(w)

        return (
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(contexts, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
        )


def collate_fn(batch):
    targets = []
    contexts = []
    weights = []
    for t, c, w in batch:
        if t.numel() > 0:
            targets.append(t)
            contexts.append(c)
            weights.append(w)

    if not targets:
        return None, None, None

    return torch.cat(targets), torch.cat(contexts), torch.cat(weights)


def train(source_filter=None, use_adv=True):
    print(f"--- Setting up ({DEVICE}) ---")
    df, unique_beatmaps, bm_to_idx, status_labels, bm_to_set_id = load_and_process_data(
        source_filter
    )

    num_items = len(unique_beatmaps)
    print(f"Vocab: {num_items} Maps")

    # Use improved dataset with dynamic sampling
    dataset = BeatmapCollectionDataset(
        df,
        bm_to_idx,
        bm_to_set_id,
        sample_rate=SAMPLE_RATE,
        max_samples=MAX_SAMPLES_PER_COLLECTION,
    )

    # Batch size is now number of collections, not number of pairs
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_COLS,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    model = Beatmap2Vec(num_items, EMBEDDING_DIM).to(DEVICE)

    status_clf = None
    optimizer_params = list(model.parameters())

    if use_adv:
        print("Adversarial Training: ENABLED")
        status_clf = StatusClassifier(EMBEDDING_DIM).to(DEVICE)
        optimizer_params += list(status_clf.parameters())
    else:
        print("Adversarial Training: DISABLED")

    optimizer = torch.optim.Adam(optimizer_params, lr=LR)
    scaler = torch.amp.GradScaler("cuda")

    print(f"--- Training (Cols/Batch: {BATCH_SIZE_COLS}) ---")

    for epoch in range(EPOCHS):
        model.train()
        if status_clf:
            status_clf.train()

        total_loss = 0
        total_adv_loss = 0
        steps = 0

        pbar = tqdm(dataloader, desc=f"Ep {epoch + 1}/{EPOCHS}", leave=False)

        for targets, contexts, weights in pbar:
            if targets is None:
                continue

            targets = targets.to(DEVICE)
            contexts = contexts.to(DEVICE)
            weights = weights.to(DEVICE)

            negatives = torch.randint(
                0, num_items, (targets.size(0), NEG_SAMPLES), device=DEVICE
            )

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                sg_loss = model(targets, contexts, negatives, weights)

                loss = sg_loss
                adv_loss_val = 0.0

                if use_adv and status_clf:
                    # Adversarial
                    item_emb = model.target_embedding.weight
                    status_preds = status_clf(item_emb).squeeze()
                    adv_loss = F.binary_cross_entropy_with_logits(
                        status_preds, status_labels
                    )
                    loss = loss + adv_loss
                    adv_loss_val = adv_loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += sg_loss.item()
            total_adv_loss += adv_loss_val
            steps += 1

            if steps % 10 == 0:
                pbar.set_postfix(
                    {"sg": f"{sg_loss.item():.2f}", "adv": f"{adv_loss_val:.2f}"}
                )

        print(
            f"Ep {epoch + 1}: SGLoss = {total_loss / steps:.4f}, AdvLoss = {total_adv_loss / steps:.4f}"
        )

    print("Saving...")
    model.eval()

    embeddings = model.get_embeddings().cpu().detach().numpy()

    pd.DataFrame(
        [
            {"beatmap_id": k, "embedding": embeddings[i].tolist()}
            for k, i in bm_to_idx.items()
        ]
    ).to_parquet(COLLECTIONS_DIR / f"beatmap2vec_embeddings_{VERSION}.parquet")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=int)
    parser.add_argument(
        "--no-adv", action="store_true", help="Disable adversarial training"
    )
    args = parser.parse_args()

    train(source_filter=args.source, use_adv=not args.no_adv)
