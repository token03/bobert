from collections import defaultdict
import math
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import time
import os
import json
import argparse
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VERSION = "v3"
MIN_MAPS_IN_COLLECTION = 10
MAX_MAPS_IN_COLLECTION = 3000
MIN_COLLECTIONS_PER_MAP = 2
JACCARD_THRESHOLD = 0.75
SONG_DAMPENING_POWER = 0.95
N_TOPICS = 144
ALPHA = 1e-4
L1_RATIO = 0.4

DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
COLLECTION_FILTER_PATH = COLLECTIONS_DIR / "collection_filter.json"
VERTEX_PATH = COLLECTIONS_DIR / "collections.parquet"
COLLECTIONS_DATA_PATH = COLLECTIONS_DIR / "collection_beatmaps.parquet"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"


class OsuNMF:
    def __init__(
        self,
        n_components: int = 128,
        max_iter: int = 500,
        tol: float = 1e-4,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        init: str = "nndsvda",
        random_state: int | None = None,
        verbose: int = 1,
        loss_check_interval: int = 20,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")

        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.loss_check_interval = loss_check_interval
        self.device = torch.device("cuda")
        self.dtype = torch.float32

        self.W = None
        self.H = None

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

    def _nndsvd_init(self, X_sparse):
        n_samples, n_features = X_sparse.shape
        k = min(self.k, min(n_samples, n_features) - 1)

        if self.verbose:
            print("Running NNDSVD initialization (CPU)...")

        U, S, Vt = svds(X_sparse.astype(np.float64), k=k)

        U, S, Vt = U[:, ::-1], S[::-1], Vt[::-1, :]

        W = np.zeros((n_samples, self.k), dtype=np.float32)
        H = np.zeros((self.k, n_features), dtype=np.float32)

        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])

        for j in range(1, k):
            u, v = U[:, j], Vt[j, :]
            u_pos, u_neg = np.maximum(u, 0), np.abs(np.minimum(u, 0))
            v_pos, v_neg = np.maximum(v, 0), np.abs(np.minimum(v, 0))

            m_pos = np.linalg.norm(u_pos) * np.linalg.norm(v_pos)
            m_neg = np.linalg.norm(u_neg) * np.linalg.norm(v_neg)

            if m_pos >= m_neg:
                W[:, j] = np.sqrt(S[j] * m_pos) * (
                    u_pos / (np.linalg.norm(u_pos) + 1e-10)
                )
                H[j, :] = np.sqrt(S[j] * m_pos) * (
                    v_pos / (np.linalg.norm(v_pos) + 1e-10)
                )
            else:
                W[:, j] = np.sqrt(S[j] * m_neg) * (
                    u_neg / (np.linalg.norm(u_neg) + 1e-10)
                )
                H[j, :] = np.sqrt(S[j] * m_neg) * (
                    v_neg / (np.linalg.norm(v_neg) + 1e-10)
                )

        avg = X_sparse.mean()
        W[W < 1e-10] = avg
        H[H < 1e-10] = avg

        return W, H

    def fit_transform(self, X_sparse):
        print(f"--- Initializing CUDA HALS NMF (Device: {self.device}) ---")
        n_samples, n_features = X_sparse.shape

        coo = X_sparse.tocoo()
        indices = torch.stack(
            [torch.from_numpy(coo.row), torch.from_numpy(coo.col)]
        ).to(self.device, dtype=torch.long)
        values = torch.from_numpy(coo.data).to(self.device, dtype=self.dtype)
        X_gpu = torch.sparse_coo_tensor(indices, values, (n_samples, n_features))

        if self.init == "nndsvda":
            W_np, H_np = self._nndsvd_init(X_sparse)
            self.W = torch.tensor(W_np, device=self.device, dtype=self.dtype)
            self.H = torch.tensor(H_np, device=self.device, dtype=self.dtype)
        else:
            self.W = torch.rand(n_samples, self.k, device=self.device, dtype=self.dtype)
            self.H = torch.rand(
                self.k, n_features, device=self.device, dtype=self.dtype
            )

        eps = 1e-16
        l1_reg = self.alpha * self.l1_ratio
        l2_reg = self.alpha * (1 - self.l1_ratio)

        if self.verbose:
            pbar = tqdm(range(self.max_iter), desc="HALS Training")
        else:
            pbar = range(self.max_iter)

        for i in pbar:
            WtW = torch.mm(self.W.t(), self.W)
            WtX = torch.sparse.mm(X_gpu.t(), self.W).t()

            for k in range(self.k):
                denom = WtW[k, k] + l2_reg + eps

                current_projection = torch.mv(self.H.t(), WtW[k])
                numerator = (
                    WtX[k] - current_projection + (WtW[k, k] * self.H[k]) - l1_reg
                )

                self.H[k] = torch.nn.functional.relu(numerator / denom)

            HHt = torch.mm(self.H, self.H.t())
            XHt = torch.sparse.mm(X_gpu, self.H.t())

            for k in range(self.k):
                denom = HHt[k, k] + l2_reg + eps

                current_projection = torch.mv(self.W, HHt[k])
                numerator = (
                    XHt[:, k] - current_projection + (HHt[k, k] * self.W[:, k]) - l1_reg
                )

                self.W[:, k] = torch.nn.functional.relu(numerator / denom)

            h_norm = torch.norm(self.H, p=2, dim=1)

            dead_indices = torch.nonzero(h_norm < 1e-10).flatten()

            if len(dead_indices) > 0:
                n_dead = len(dead_indices)

                random_row_indices = torch.randint(
                    0, n_samples, (n_dead,), device=self.device
                )

                target_X = X_gpu.index_select(0, random_row_indices).to_dense()

                current_prediction = torch.mm(self.W[random_row_indices], self.H)

                residual = torch.nn.functional.relu(target_X - current_prediction)

                mask_empty = (residual.sum(dim=1) < 1e-10).unsqueeze(1)
                final_revival = torch.where(mask_empty, target_X, residual)

                self.H[dead_indices, :] = final_revival

                self.H[dead_indices, :] += (
                    torch.rand(
                        (n_dead, n_features), device=self.device, dtype=self.dtype
                    )
                    * 1e-6
                )

                w_avg = self.W.mean().item()
                self.W[:, dead_indices] = torch.full(
                    (n_samples, n_dead), w_avg, device=self.device, dtype=self.dtype
                )

                h_norm = torch.norm(self.H, p=2, dim=1)

            h_norm = h_norm + eps
            self.H /= h_norm.unsqueeze(1)
            self.W *= h_norm

            if i % self.loss_check_interval == 0:
                pass

        print("Done. Copying to CPU...")
        return self.W.cpu().numpy()

    @property
    def components_(self):
        return self.H.cpu().numpy()


def get_collection_names(collection_tuples):
    if not os.path.exists(VERTEX_PATH):
        print(f"Warning: {VERTEX_PATH} not found, using 'Unknown' for all names")
        return {key: "Unknown" for key in collection_tuples}

    vertex_df = pd.read_parquet(VERTEX_PATH)

    collection_ids = [cid for cid, _ in collection_tuples]
    sources = [src for _, src in collection_tuples]

    vertex_df = vertex_df[
        (vertex_df["collection_id"].isin(collection_ids))
        & (vertex_df["source"].isin(sources))
    ]

    name_map = {}
    for _, row in vertex_df.iterrows():
        key = (row["collection_id"], row["source"])
        name = row.get("name", "Unknown")
        if pd.isna(name):
            name = "Unknown"
        name_map[key] = name

    for key in collection_tuples:
        if key not in name_map:
            name_map[key] = "Unknown"

    return name_map


def deduplicate_collections(df, threshold, probe_items=32):
    print("--- Starting Deduplication Process ---")
    start_time = time.time()

    print("Grouping collections...")
    df["collection_key"] = list(zip(df["collection_id"], df["source"]))
    col_groups = df.groupby("collection_key")["beatmap_id"].apply(set).to_dict()

    print("Step 1: Removing Exact Duplicates...")
    content_hashes = {}
    for ckey, bms in col_groups.items():
        sig = tuple(sorted(bms))
        prev = content_hashes.get(sig)
        if prev is None or ckey[0] < prev[0]:
            content_hashes[sig] = ckey

    unique_content_ckeys = set(content_hashes.values())
    print(
        f"Reduced from {len(col_groups)} to {len(unique_content_ckeys)} unique content sets."
    )

    print(f"Step 2: Fuzzy Deduplication (Threshold: {threshold})...")

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
            )

            for kept_ckey, approx_overlap in candidates:
                B = kept_sets[kept_ckey]
                len_b = len(B)

                if len_a < threshold * len_b or len_b < threshold * len_a:
                    continue

                req = min_required_overlap(len_a, len_b, threshold)

                if approx_overlap == 0:
                    continue

                inter = len(A & B)
                if inter < req:
                    continue

                union = len_a + len_b - inter
                jacc = inter / union
                if jacc >= threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            kept_ckeys.append(ckey)
            kept_sets[ckey] = A
            for bm in A:
                postings[bm].append(ckey)

    print(
        f"Final Collection Count: {len(kept_ckeys)} (Removed {len(col_groups) - len(kept_ckeys)} duplicates)"
    )
    print(f"Deduplication took {(time.time() - start_time):.2f}s")

    return df[df["collection_key"].isin(kept_ckeys)].copy()


def run_nmf(source_filter=None):
    print("--- Loading Data ---")
    df = pd.read_parquet(COLLECTIONS_DATA_PATH)

    if source_filter is not None:
        print(f"--- Filtering by Source: {source_filter} ---")
        df = df[df["source"] == source_filter].copy()
        print(f"Kept {len(df)} rows from source {source_filter}")

    if os.path.exists(COLLECTION_FILTER_PATH):
        print("--- Applying Collection and User Filters ---")
        with open(COLLECTION_FILTER_PATH, "r") as f:
            filter_data = json.load(f)

        if "collections" in filter_data:
            collections_to_remove = []
            for source, collection_ids in filter_data["collections"].items():
                collections_to_remove.extend(collection_ids)

            if collections_to_remove:
                initial_count = len(df)
                df = df[~df["collection_id"].isin(collections_to_remove)].copy()
                print(
                    f"Removed {initial_count - len(df)} rows from {len(collections_to_remove)} filtered collections"
                )

        if "users" in filter_data:
            if os.path.exists(VERTEX_PATH):
                vertex_df = pd.read_parquet(VERTEX_PATH)

                users_to_remove = []
                for source, user_ids in filter_data["users"].items():
                    users_to_remove.extend(user_ids)

                if users_to_remove:
                    filtered_collections = vertex_df[
                        vertex_df["uploader_id"].isin(users_to_remove)
                    ]["collection_id"].unique()

                    initial_count = len(df)
                    df = df[~df["collection_id"].isin(filtered_collections)].copy()
                    print(
                        f"Removed {initial_count - len(df)} rows from {len(filtered_collections)} collections by {len(set(users_to_remove))} filtered users"
                    )
            else:
                print(f"Warning: {VERTEX_PATH} not found, skipping user filter")

    print("--- Loading Beatmap Metadata ---")
    beatmaps_df = pd.read_parquet(
        BEATMAPS_PATH, columns=["id", "beatmapset_id", "title", "mode"]
    )
    beatmaps_df = beatmaps_df.rename(columns={"id": "beatmap_id"})
    df = df.merge(beatmaps_df, on="beatmap_id", how="left")

    n_missing = df[df["beatmapset_id"].isna()]["beatmap_id"].nunique()
    print(f"Unique beatmaps missing metadata: {n_missing}")

    print("--- Filtering by Game Mode ---")
    collection_mode_stats = df.groupby("collection_id").apply(
        lambda x: (x["mode"] != "osu").sum() / len(x) if len(x) > 0 else 0,
        include_groups=False,
    )
    collections_to_keep = collection_mode_stats[collection_mode_stats <= 0.5].index
    df = df[df["collection_id"].isin(collections_to_keep)].copy()
    df = df[df["mode"] == "osu"].copy()

    col_counts = df.groupby("collection_id")["beatmap_id"].count()
    valid_collections = col_counts[
        (col_counts >= MIN_MAPS_IN_COLLECTION) & (col_counts <= MAX_MAPS_IN_COLLECTION)
    ].index
    df = df[df["collection_id"].isin(valid_collections)].copy()

    df = deduplicate_collections(df, JACCARD_THRESHOLD)

    bm_counts = df["beatmap_id"].value_counts()
    valid_maps = bm_counts[(bm_counts >= MIN_COLLECTIONS_PER_MAP)].index
    df = df[df["beatmap_id"].isin(valid_maps)].copy()

    missing_metadata = df["beatmapset_id"].isna().sum()
    if missing_metadata > 0:
        print(f"Dropping {missing_metadata} rows with missing beatmap metadata...")
        df = df.dropna(subset=["beatmapset_id"]).copy()

    print("--- Applying Set/Song Dampening ---")
    df["song_occurrence"] = df.groupby(["collection_id", "beatmapset_id"])[
        "beatmap_id"
    ].transform("count")
    df["weight"] = 1.0 / (df["song_occurrence"] ** SONG_DAMPENING_POWER)

    df["collection_key"] = list(zip(df["collection_id"], df["source"]))
    unique_collections = sorted(df["collection_key"].unique())
    unique_beatmaps = sorted(df["beatmap_id"].unique())

    col_to_idx = {ckey: i for i, ckey in enumerate(unique_collections)}
    bm_to_idx = {bid: i for i, bid in enumerate(unique_beatmaps)}
    idx_to_bm = {i: bid for bid, i in bm_to_idx.items()}
    idx_to_col = {i: ckey for ckey, i in col_to_idx.items()}

    beatmap_to_beatmapset = (
        df.drop_duplicates("beatmap_id")
        .set_index("beatmap_id")["beatmapset_id"]
        .to_dict()
    )
    beatmapset_to_title = (
        df.drop_duplicates("beatmapset_id")
        .set_index("beatmapset_id")["title"]
        .to_dict()
    )

    print(f"Matrix: {len(unique_collections)} Col x {len(unique_beatmaps)} Maps")

    row = df["collection_key"].map(col_to_idx).values
    col = df["beatmap_id"].map(bm_to_idx).values
    data = df["weight"].values

    X = csr_matrix(
        (data, (row, col)), shape=(len(unique_collections), len(unique_beatmaps))
    )

    print("Applying TF-IDF (L2 Norm)...")
    tfidf = TfidfTransformer(
        norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False
    )
    X_tfidf = tfidf.fit_transform(X)

    print(f"Running CUDA HALS NMF ({N_TOPICS} topics)...")
    nmf = OsuNMF(
        n_components=N_TOPICS,
        max_iter=1000,
        tol=1e-5,
        alpha=ALPHA,
        l1_ratio=L1_RATIO,
        init="nndsvda",
        random_state=42,
        verbose=1,
    )

    start = time.time()
    W = nmf.fit_transform(X_tfidf)
    H = nmf.components_
    print(f"Completed in {(time.time() - start) / 60:.2f} minutes")

    print("Calculating topic statistics...")
    col_counts = pd.Series(W.argmax(axis=1)).value_counts()
    bm_counts = pd.Series(H.argmax(axis=0)).value_counts()

    stats_df = pd.DataFrame(
        [
            {
                "topic_id": i,
                "collections": int(col_counts.get(i, 0)),
                "beatmaps": int(bm_counts.get(i, 0)),
            }
            for i in range(N_TOPICS)
        ]
    )

    print("\nTopic Distribution Statistics:")
    print(stats_df.to_string(index=False))

    print("Saving weights as parquet...")
    h_records = []
    for t_idx in range(N_TOPICS):
        relevant_b_indices = np.where(H[t_idx] > 1e-4)[0]
        for b_idx in relevant_b_indices:
            h_records.append(
                {
                    "beatmap_id": unique_beatmaps[b_idx],
                    "topic_id": t_idx,
                    "weight": float(H[t_idx, b_idx]),
                }
            )
    pd.DataFrame(h_records).to_parquet(
        COLLECTIONS_DIR / f"beatmap_topic_weights_{VERSION}.parquet"
    )

    w_records = []
    for c_idx in range(len(unique_collections)):
        relevant_t_indices = np.where(W[c_idx] > 1e-4)[0]
        collection_id, source = unique_collections[c_idx]
        for t_idx in relevant_t_indices:
            w_records.append(
                {
                    "collection_id": collection_id,
                    "source": source,
                    "topic_id": t_idx,
                    "weight": float(W[c_idx, t_idx]),
                }
            )
    pd.DataFrame(w_records).to_parquet(
        COLLECTIONS_DIR / f"collection_topic_weights_{VERSION}.parquet"
    )

    print("Generating summaries...")

    map_topic_counts = {}
    for t_idx in range(N_TOPICS):
        top_indices = H[t_idx].argsort()[::-1][:30]
        map_topic_counts.update(
            {idx: map_topic_counts.get(idx, 0) + 1 for idx in top_indices}
        )

    summary = []
    for t_idx in range(N_TOPICS):
        top_indices = H[t_idx].argsort()[::-1]

        song_groups = {}
        seen_songs = set()

        for i in top_indices:
            if len(seen_songs) >= 20:
                break

            bid = idx_to_bm[i]
            bset_id = beatmap_to_beatmapset.get(bid)

            if bset_id is None:
                continue

            if bset_id not in seen_songs:
                seen_songs.add(bset_id)
                song_groups[bset_id] = []

            song_groups[bset_id].append(bid)

        song_displays = []
        for bset_id, bids in song_groups.items():
            title = beatmapset_to_title.get(bset_id, "Unknown")
            title = title.replace("|", "-").replace(",", " ")
            bids_str = ", ".join(map(str, bids))

            is_generic = any(
                map_topic_counts.get(bm_to_idx.get(bid, -1), 0) > (N_TOPICS * 0.10)
                for bid in bids
            )

            display = (
                f"*{title}* ({bids_str})" if is_generic else f"{title} ({bids_str})"
            )
            song_displays.append(display)

        summary.append({"topic_id": t_idx, "top_maps": " | ".join(song_displays)})
    summary_df = pd.DataFrame(summary)

    all_top_ckeys = set()
    top_ckeys_per_topic = []
    for t_idx in range(N_TOPICS):
        top_indices = W[:, t_idx].argsort()[::-1][:10]
        ckeys = [unique_collections[i] for i in top_indices]
        top_ckeys_per_topic.append(ckeys)
        all_top_ckeys.update(ckeys)

    col_names_map = get_collection_names(list(all_top_ckeys))
    col_summary = []
    for t_idx in range(N_TOPICS):
        ckeys = top_ckeys_per_topic[t_idx]
        names = [
            f"{col_names_map.get(ckey, 'Unknown')} ({ckey[0]}, src={ckey[1]})"
            for ckey in ckeys
        ]
        col_summary.append({"topic_id": t_idx, "top_collections": " | ".join(names)})
    col_summary_df = pd.DataFrame(col_summary)

    final_summary = stats_df.merge(col_summary_df, on="topic_id", how="left").merge(
        summary_df, on="topic_id", how="left"
    )

    final_summary.columns = [
        "topic_id",
        "collection_count",
        "beatmap_count",
        "collection_names",
        "beatmap_names",
    ]
    final_summary.to_csv(COLLECTIONS_DIR / f"topic_summary_{VERSION}.csv", index=False)
    print(f"Fused summary saved to {COLLECTIONS_DIR / f'topic_summary_{VERSION}.csv'}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NMF topic modeling on osu! collections"
    )
    parser.add_argument(
        "-s",
        "--source",
        type=int,
        choices=[1, 2],
        help="Filter by source: 1=OsuCollector, 2=OsuStats (default: use all sources)",
    )
    args = parser.parse_args()

    run_nmf(source_filter=args.source)
