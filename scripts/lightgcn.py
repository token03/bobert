import sys
import os
import math
import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


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
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(self.grl(x))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
COLLECTIONS_DATA_PATH = COLLECTIONS_DIR / "collection_beatmaps.parquet"
VERTEX_PATH = COLLECTIONS_DIR / "collections.parquet"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
COLLECTION_FILTER_PATH = COLLECTIONS_DIR / "collection_filter.json"

VERSION = "v1"
MIN_MAPS_IN_COLLECTION = 10
MAX_MAPS_IN_COLLECTION = 3000
MIN_COLLECTIONS_PER_MAP = 2
JACCARD_THRESHOLD = 0.75
SONG_DAMPENING_POWER = 0.95

EMBEDDING_DIM = 64
NUM_LAYERS = 3
BATCH_SIZE = 131072
LR = 0.001
EPOCHS = 30
CL_RATE = 0.2
EPS = 0.2
TEMP = 0.2
ADV_LAMBDA = 0.1
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

    def min_required_overlap(len_a, len_b, t):
        return math.ceil((t * (len_a + len_b)) / (1.0 + t))

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
    col_to_idx = {ckey: i for i, ckey in enumerate(unique_collections)}
    bm_to_idx = {bid: i for i, bid in enumerate(unique_beatmaps)}

    status_map = df.groupby("beatmap_id")["status"].first()
    status_labels = torch.zeros(
        len(unique_beatmaps), dtype=torch.float32, device=DEVICE
    )
    for bid, idx in bm_to_idx.items():
        if bid in status_map:
            status = status_map[bid]
            status_labels[idx] = 1.0 if status in ['1', '2', '3', '4'] else 0.0

    return df, unique_collections, unique_beatmaps, col_to_idx, bm_to_idx, status_labels


def compute_normalized_laplacian(
    user_indices, item_indices, values, num_users, num_items
):
    num_nodes = num_users + num_items

    row = torch.cat([user_indices, item_indices + num_users])
    col = torch.cat([item_indices + num_users, user_indices])
    vals = torch.cat([values, values])

    deg = torch.zeros(num_nodes, device=DEVICE)
    deg.scatter_add_(0, row, vals)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    norm_vals = vals * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    indices = torch.stack([row, col], dim=0)

    return torch.sparse_coo_tensor(
        indices, norm_vals, (num_nodes, num_nodes)
    ).coalesce()


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        num_layers,
    ):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, sparse_adj):
        with torch.amp.autocast("cuda", enabled=False):
            ego_embeddings = torch.cat(
                [self.user_embedding.weight, self.item_embedding.weight], dim=0
            ).float()
            all_embeddings = [ego_embeddings]

            for _ in range(self.num_layers):
                ego_embeddings = torch.sparse.mm(sparse_adj, ego_embeddings)
                all_embeddings.append(ego_embeddings)

            final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        return final_embeddings


def train(source_filter=None):
    print(f"--- Setting up ({DEVICE}) ---")
    df, unique_collections, unique_beatmaps, col_to_idx, bm_to_idx, status_labels = (
        load_and_process_data(source_filter)
    )

    num_users = len(unique_collections)
    num_items = len(unique_beatmaps)
    print(f"Graph: {num_users} Cols, {num_items} Maps, {len(df)} Edges")

    user_indices = torch.tensor(
        df["collection_key"].map(col_to_idx).values, dtype=torch.long, device=DEVICE
    )
    item_indices = torch.tensor(
        df["beatmap_id"].map(bm_to_idx).values, dtype=torch.long, device=DEVICE
    )
    weights = torch.tensor(df["weight"].values, dtype=torch.float32, device=DEVICE)


    print("Building Sparse Adjacency Matrix...")
    sparse_adj = compute_normalized_laplacian(
        user_indices, item_indices, weights, num_users, num_items
    )

    model = LightGCN(
        num_users,
        num_items,
        EMBEDDING_DIM,
        NUM_LAYERS,
    ).to(DEVICE)
    status_clf = StatusClassifier(EMBEDDING_DIM).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(status_clf.parameters()), lr=0.005
    )
    scaler = torch.amp.GradScaler("cuda")

    print(f"--- Training (Batch Size: {BATCH_SIZE}) ---")
    total_edges = len(df)
    perm_indices = torch.arange(total_edges, device=DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        status_clf.train()
        total_loss = 0
        total_adv_loss = 0

        perm = perm_indices[torch.randperm(total_edges)]
        num_batches = (total_edges + BATCH_SIZE - 1) // BATCH_SIZE

        pbar = tqdm(range(num_batches), desc=f"Ep {epoch + 1}/{EPOCHS}", leave=False)

        for i in pbar:
            optimizer.zero_grad()

            idx = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            batch_users = user_indices[idx]
            batch_pos = item_indices[idx] + num_users
            batch_neg = (
                torch.randint(0, num_items, (len(idx),), device=DEVICE) + num_users
            )

            with torch.amp.autocast("cuda"):
                all_emb = model(sparse_adj)

                u_emb = all_emb[batch_users]
                p_emb = all_emb[batch_pos]
                n_emb = all_emb[batch_neg]

                pos_scores = (u_emb * p_emb).sum(dim=1)
                neg_scores = (u_emb * n_emb).sum(dim=1)

                reg_loss = (
                    (1 / 2)
                    * (
                        u_emb.norm(2).pow(2)
                        + p_emb.norm(2).pow(2)
                        + n_emb.norm(2).pow(2)
                    )
                    / float(len(idx))
                )
                rec_loss = F.softplus(neg_scores - pos_scores).mean() + (
                    1e-4 * reg_loss
                )

                item_emb = all_emb[num_users:]
                status_preds = status_clf(item_emb).squeeze()
                adv_loss = F.binary_cross_entropy_with_logits(
                    status_preds, status_labels
                )

                loss = rec_loss + adv_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += rec_loss.item()
            total_adv_loss += adv_loss.item()

        print(
            f"Ep {epoch + 1}: RecLoss = {total_loss / num_batches:.4f}, AdvLoss = {total_adv_loss / num_batches:.4f}"
        )

    print("Saving...")
    model.eval()
    with torch.no_grad():
        final_emb = model(sparse_adj)

    user_emb_np = final_emb[:num_users].cpu().numpy()
    item_emb_np = final_emb[num_users:].cpu().numpy()

    pd.DataFrame(
        [
            {
                "collection_id": k[0],
                "source": k[1],
                "embedding": user_emb_np[i].tolist(),
            }
            for k, i in col_to_idx.items()
        ]
    ).to_parquet(COLLECTIONS_DIR / f"collection_embeddings_{VERSION}.parquet")

    pd.DataFrame(
        [
            {"beatmap_id": k, "embedding": item_emb_np[i].tolist()}
            for k, i in bm_to_idx.items()
        ]
    ).to_parquet(COLLECTIONS_DIR / f"beatmap_embeddings_{VERSION}.parquet")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=int)
    args = parser.parse_args()
    train(source_filter=args.source)