import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from core.data.mining import _whiten_centered_rows
from scripts.common.collections import deduplicate_collections
from scripts.common.paths import BEATMAPS_PATH, COLLECTIONS_DIR, DATA_DIR


warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state.*",
    category=UserWarning,
)


COLLECTION_EDGES_PATH = COLLECTIONS_DIR / "edges.parquet"
COLLECTION_FILTER_PATH = COLLECTIONS_DIR / "collection_filter.json"
GRAPH_EMBEDDINGS_PATH = DATA_DIR / "graph.parquet"
PRETRAIN_EMBEDDINGS_PATH = DATA_DIR / "embeddings-pretrain.parquet"

MIN_MAPS_IN_COLLECTION = 15
MIN_COLLECTIONS_PER_MAP = 2
JACCARD_THRESHOLD = 0.9
MIN_COLLECTION_QUALITY = 0.10
NULL95_SIZES = np.array([5, 6, 8, 10, 15, 20, 30, 50, 100, 250, 500, 1000, 2500])
NULL95_TRIALS = 1000
NULL95_SEED = 13

EMBEDDING_DIM = 64
NUM_LAYERS = 2
LAYER_CL = 1
EPS = 0.10
TAU = 0.20
CL_WEIGHT = 0.10
L2_REG = 1e-6

BPR_BATCH_SIZE = 65536
CL_MAX_NODES = 2048
CL_EVERY = 5
TOTAL_STEPS = 2000
LOG_EVERY = 100

LR = 2e-3
GRAD_CLIP_NORM = 5.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_process_data(source_filter=None):
    df = pd.read_parquet(COLLECTION_EDGES_PATH)

    if source_filter:
        df = df[df["source"] == source_filter].copy()

    df["collection_key"] = list(zip(df["collection_id"], df["source"]))

    if os.path.exists(COLLECTION_FILTER_PATH):
        with open(COLLECTION_FILTER_PATH, "r") as f:
            filter_data = json.load(f)
        if "collections" in filter_data:
            bad_collection_keys = set()
            for src, ids in filter_data["collections"].items():
                src_id = int(src)
                bad_collection_keys.update((cid, src_id) for cid in ids)
            if bad_collection_keys:
                key_series = pd.Series(
                    list(zip(df["collection_id"], df["source"])), index=df.index
                )
                df = df[~key_series.isin(bad_collection_keys)].copy()

    beatmaps_df = pd.read_parquet(
        BEATMAPS_PATH,
        columns=["id", "beatmapset_id", "mode"],
    ).rename(columns={"id": "beatmap_id"})

    df = df.merge(beatmaps_df, on="beatmap_id", how="left")
    df = df[df["mode"] == "osu"].copy()
    df = df.dropna(subset=["beatmapset_id"]).copy()
    pretrain_ids = set(
        pl.read_parquet(PRETRAIN_EMBEDDINGS_PATH, columns=["beatmap_id"])["beatmap_id"]
        .cast(pl.Int64)
        .to_list()
    )
    df = df[df["beatmap_id"].astype(int).isin(pretrain_ids)].copy()

    col_counts = df.groupby("collection_key")["beatmap_id"].count()
    valid_cols = col_counts[col_counts >= MIN_MAPS_IN_COLLECTION].index
    df = df[df["collection_key"].isin(valid_cols)].copy()

    df = deduplicate_collections(df, JACCARD_THRESHOLD)
    df = df.drop_duplicates(["collection_key", "beatmap_id"]).copy()
    collection_quality, null95_q = collection_reliability(df)
    collection_sizes = df.groupby("collection_key")["beatmap_id"].count().to_dict()
    valid_cols = [
        ckey
        for ckey, q in collection_quality.items()
        if collection_sizes[ckey] >= MIN_MAPS_IN_COLLECTION
        and q >= max(MIN_COLLECTION_QUALITY, collection_null95(collection_sizes[ckey], null95_q))
    ]
    df = df[df["collection_key"].isin(valid_cols)].copy()

    df = prune_graph_edges(df)
    collection_quality = {
        ckey: collection_quality[ckey] for ckey in df["collection_key"].unique()
    }

    unique_collections = sorted(df["collection_key"].unique())
    unique_beatmaps = sorted(df["beatmap_id"].unique())
    col_to_idx = {ckey: i for i, ckey in enumerate(unique_collections)}
    bm_to_idx = {bid: i for i, bid in enumerate(unique_beatmaps)}

    return df, unique_collections, unique_beatmaps, col_to_idx, bm_to_idx, collection_quality


def collection_null95(size, null95_q):
    return float(np.interp(np.log(size), np.log(NULL95_SIZES), null95_q))


def prune_graph_edges(df):
    while True:
        before = len(df)

        col_counts = df.groupby("collection_key")["beatmap_id"].count()
        valid_cols = col_counts[col_counts >= MIN_MAPS_IN_COLLECTION].index
        df = df[df["collection_key"].isin(valid_cols)].copy()

        bm_counts = df["beatmap_id"].value_counts()
        valid_maps = bm_counts[bm_counts >= MIN_COLLECTIONS_PER_MAP].index
        df = df[df["beatmap_id"].isin(valid_maps)].copy()

        if len(df) == before:
            return df


def compute_weighted_normalized_adj(user_indices, item_indices, edge_weights, num_users, num_items):
    num_nodes = num_users + num_items

    row = torch.cat([user_indices, item_indices + num_users])
    col = torch.cat([item_indices + num_users, user_indices])
    vals = torch.cat([edge_weights, edge_weights]).float()

    deg = torch.zeros(num_nodes, device=row.device, dtype=torch.float32)
    deg.scatter_add_(0, row, vals)

    deg_inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
    norm_vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
        vals * norm_vals,
        size=(num_nodes, num_nodes),
        device=row.device,
        check_invariants=False,
    ).coalesce()
    if row.device.type == "cuda":
        adj = adj.to_sparse_csr()
    return adj


def collection_reliability(df):
    pretrain = pd.read_parquet(PRETRAIN_EMBEDDINGS_PATH, columns=["beatmap_id", "embedding"])
    emb = torch.from_numpy(
        _whiten_centered_rows(
            np.stack(pretrain["embedding"].to_numpy()).astype("float32")
        )
    )
    null95_q = compute_null95_q(emb)
    emb_lookup = dict(zip(pretrain["beatmap_id"].astype(int), emb))

    values = []
    for collection_key, group in df.groupby("collection_key"):
        rows = [emb_lookup[int(bid)] for bid in group["beatmap_id"]]
        x = torch.stack(rows, dim=0)
        r = float(x.mean(dim=0).norm())
        mu = len(rows) ** -0.5
        q = max(0.0, min(1.0, (r - mu) / max(1.0 - mu, 1e-6)))
        values.append((collection_key, q))
    return dict(values), null95_q


def compute_null95_q(emb):
    generator = torch.Generator().manual_seed(NULL95_SEED)
    null95_q = []
    for size in NULL95_SIZES:
        values = []
        for _ in range(NULL95_TRIALS):
            idx = torch.randint(0, emb.shape[0], (int(size),), generator=generator)
            r = float(emb[idx].mean(dim=0).norm())
            mu = int(size) ** -0.5
            values.append(max(0.0, min(1.0, (r - mu) / max(1.0 - mu, 1e-6))))
        null95_q.append(float(np.quantile(values, 0.95)))
    return np.array(null95_q)


def build_collection_items(df, col_to_idx, bm_to_idx, num_users):
    items = [[] for _ in range(num_users)]
    for collection_key, beatmap_id in zip(df["collection_key"], df["beatmap_id"]):
        items[col_to_idx[collection_key]].append(bm_to_idx[int(beatmap_id)])
    lengths = torch.tensor([len(v) for v in items], dtype=torch.long, device=DEVICE)
    offsets = torch.empty(num_users, dtype=torch.long, device=DEVICE)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths[:-1], dim=0)
    flat = torch.tensor([item for group in items for item in group], dtype=torch.long, device=DEVICE)
    return flat, offsets, lengths


def build_degree_sampler(item_degrees):
    max_degree = int(item_degrees.max().item())
    degree_counts = torch.bincount(item_degrees, minlength=max_degree + 1)
    degree_offsets = torch.empty(max_degree + 2, dtype=torch.long, device=DEVICE)
    degree_offsets[0] = 0
    degree_offsets[1:] = torch.cumsum(degree_counts, dim=0)
    degree_items = torch.argsort(item_degrees)
    degree_positions = torch.empty_like(degree_items)
    degree_positions[degree_items] = torch.arange(degree_items.numel(), device=DEVICE)
    return degree_items, degree_offsets, degree_positions


def sample_bpr_batch(
    collection_items,
    collection_offsets,
    collection_lengths,
    collection_weights,
    item_degrees,
    degree_items,
    degree_offsets,
    degree_positions,
):
    batch_users = torch.multinomial(collection_weights, BPR_BATCH_SIZE, replacement=True)
    local = (torch.rand(BPR_BATCH_SIZE, device=DEVICE) * collection_lengths[batch_users]).long()
    batch_pos = collection_items[collection_offsets[batch_users] + local]

    degrees = item_degrees[batch_pos]
    starts = degree_offsets[degrees]
    lengths = degree_offsets[degrees + 1] - starts
    local_pos = degree_positions[batch_pos] - starts
    batch_neg = torch.empty_like(batch_pos)

    multiple = lengths > 1
    if multiple.any():
        local_neg = (
            torch.rand(int(multiple.sum()), device=DEVICE) * (lengths[multiple] - 1)
        ).long()
        local_neg += (local_neg >= local_pos[multiple]).long()
        batch_neg[multiple] = degree_items[starts[multiple] + local_neg]

    singleton = ~multiple
    if singleton.any():
        singleton_neg = torch.randint(
            0, item_degrees.numel() - 1, (int(singleton.sum()),), device=DEVICE
        )
        singleton_neg += (singleton_neg >= batch_pos[singleton]).long()
        batch_neg[singleton] = singleton_neg
    return batch_users, batch_pos, batch_neg


class XSimGCL(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, layer_cl, eps):
        super().__init__()
        if layer_cl < 1 or layer_cl > num_layers:
            raise ValueError("layer_cl must be in [1, num_layers]")
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.layer_cl = layer_cl
        self.eps = eps
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, sparse_adj, perturbed=False):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_outputs = [x]
        cl_output = None

        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            for layer_idx in range(self.num_layers):
                x = torch.sparse.mm(sparse_adj, x)
                if perturbed:
                    noise = F.normalize(torch.rand_like(x), dim=-1)
                    x = x + torch.sign(x) * noise * self.eps
                layer_outputs.append(x)
                if layer_idx + 1 == self.layer_cl:
                    cl_output = x
            final = torch.stack(layer_outputs, dim=0).mean(dim=0)

        user_final, item_final = torch.split(final, [self.num_users, self.num_items])
        if not perturbed:
            return user_final, item_final

        user_cl, item_cl = torch.split(cl_output, [self.num_users, self.num_items])
        return user_final, item_final, user_cl, item_cl


def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)
    return F.softplus(neg_scores - pos_scores).mean()


def info_nce(z1, z2, temperature):
    if z1.shape[0] < 2:
        return z1.new_zeros(())
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def sample_unique(values, max_count):
    values = torch.unique(values)
    if values.numel() > max_count:
        values = values[torch.randperm(values.numel(), device=values.device)[:max_count]]
    return values


def train(source_filter=None):
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print(f"--- Setting up ({DEVICE}) ---")
    df, unique_collections, unique_beatmaps, col_to_idx, bm_to_idx, collection_quality = (
        load_and_process_data(source_filter)
    )
    num_users = len(unique_collections)
    num_items = len(unique_beatmaps)
    print(f"Graph: {num_users} collections, {num_items} maps, {len(df)} edges")

    user_indices = torch.tensor(
        df["collection_key"].map(col_to_idx).values, dtype=torch.long, device=DEVICE
    )
    item_indices = torch.tensor(
        df["beatmap_id"].map(bm_to_idx).values, dtype=torch.long, device=DEVICE
    )
    edge_weights = torch.tensor(
        [
            max(collection_quality[ckey], 1e-6)
            for ckey in df["collection_key"]
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    collection_weights = torch.tensor(
        [max(collection_quality[ckey], 1e-6) for ckey in unique_collections],
        dtype=torch.float32,
        device=DEVICE,
    )
    collection_weights = collection_weights / collection_weights.sum()
    collection_items, collection_offsets, collection_lengths = build_collection_items(
        df, col_to_idx, bm_to_idx, num_users
    )
    item_degrees = torch.bincount(item_indices, minlength=num_items).clamp_min(1)
    degree_items, degree_offsets, degree_positions = build_degree_sampler(item_degrees)

    print("Building coherence-weighted normalized adjacency...")
    sparse_adj = compute_weighted_normalized_adj(
        user_indices, item_indices, edge_weights, num_users, num_items
    )

    model = XSimGCL(
        num_users, num_items, EMBEDDING_DIM, NUM_LAYERS, LAYER_CL, EPS
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    print(
        f"--- Training XSimGCL: steps={TOTAL_STEPS} (~{TOTAL_STEPS * BPR_BATCH_SIZE / len(df):.1f} edge epochs), "
        f"bpr_batch={BPR_BATCH_SIZE}, cl_max_nodes={CL_MAX_NODES} ---"
    )
    pbar = tqdm(range(1, TOTAL_STEPS + 1), desc="XSimGCL", dynamic_ncols=True)
    loss_ema = rec_ema = cl_ema = None

    for step in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        batch_users, batch_pos, batch_neg = sample_bpr_batch(
            collection_items,
            collection_offsets,
            collection_lengths,
            collection_weights,
            item_degrees,
            degree_items,
            degree_offsets,
            degree_positions,
        )

        if CL_WEIGHT > 0 and step % CL_EVERY == 0:
            user_emb, item_emb, user_cl, item_cl = model(sparse_adj, perturbed=True)

            cl_users = sample_unique(batch_users, CL_MAX_NODES)
            cl_items = sample_unique(batch_pos, CL_MAX_NODES)
            cl_loss = info_nce(user_emb[cl_users], user_cl[cl_users], TAU) + info_nce(
                item_emb[cl_items], item_cl[cl_items], TAU
            )
        else:
            user_emb, item_emb = model(sparse_adj, perturbed=False)
            cl_loss = user_emb.new_zeros(())

        rec_loss = bpr_loss(user_emb[batch_users], item_emb[batch_pos], item_emb[batch_neg])

        reg = (
            model.user_embedding(batch_users).pow(2).sum(dim=-1).mean()
            + model.item_embedding(batch_pos).pow(2).sum(dim=-1).mean()
            + model.item_embedding(batch_neg).pow(2).sum(dim=-1).mean()
        ) / 3.0
        loss = rec_loss + CL_WEIGHT * cl_loss + L2_REG * reg
        loss.backward()

        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        loss_val = float(loss.detach())
        rec_val = float(rec_loss.detach())
        cl_val = float(cl_loss.detach())
        if loss_ema is None:
            loss_ema, rec_ema, cl_ema = loss_val, rec_val, cl_val
        else:
            decay = 0.98
            loss_ema = decay * loss_ema + (1.0 - decay) * loss_val
            rec_ema = decay * rec_ema + (1.0 - decay) * rec_val
            cl_ema = decay * cl_ema + (1.0 - decay) * cl_val

        if step % LOG_EVERY == 0:
            pbar.set_postfix(
                loss=f"{loss_ema:.4f}", rec=f"{rec_ema:.4f}", cl=f"{cl_ema:.4f}"
            )

    print("Saving normalized graph embeddings...")
    model.eval()
    with torch.no_grad():
        _, item_emb = model(sparse_adj, perturbed=False)
        item_emb = F.normalize(item_emb, dim=-1)

    item_emb_np = _whiten_centered_rows(item_emb.cpu().numpy())
    pd.DataFrame(
        [
            {"beatmap_id": beatmap_id, "embedding": item_emb_np[idx].tolist()}
            for beatmap_id, idx in bm_to_idx.items()
        ]
    ).to_parquet(GRAPH_EMBEDDINGS_PATH)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=int)
    args = parser.parse_args()
    train(source_filter=args.source)
