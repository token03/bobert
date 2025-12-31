from collections import defaultdict
import math
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
import time
import requests
import os
import json
import re
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VERSION = "v5"
MIN_MAPS_IN_COLLECTION = 10
MAX_MAPS_IN_COLLECTION = 3000
MAX_FREQUENCY_PER_MAP = 0.05
MIN_COLLECTIONS_PER_MAP = 2
JACCARD_THRESHOLD = 0.90
SONG_DAMPENING_POWER = 0.9
N_TOPICS = 150
ALPHA = 0.00005
SCALING_FACTOR = 100
L1_RATIO = 0.5

DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
COLLECTIONS_DATA_PATH = COLLECTIONS_DIR / "collections_data.parquet"

def get_collection_names(collection_ids):
    cache_file = DATA_DIR / "collection_names_cache.json"
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)

    to_fetch = [cid for cid in collection_ids if str(cid) not in cache]
    if to_fetch:
        print(f"Fetching {len(to_fetch)} collection names...")
        for cid in tqdm(to_fetch, desc="Fetching names"):
            try:
                response = requests.get(f"https://osucollector.com/api/collections/{cid}", timeout=10)
                if response.status_code == 200:
                    cache[str(cid)] = response.json().get("name", "Unknown")
                else:
                    cache[str(cid)] = "Unknown"
                time.sleep(0.2)
            except Exception as e:
                print(f"Error fetching {cid}: {e}")
                cache[str(cid)] = "Unknown"
        
        with open(cache_file, "w") as f:
            json.dump(cache, f)
            
    return {int(cid): cache.get(str(cid), "Unknown") for cid in collection_ids}


def deduplicate_collections(df, threshold=0.90, probe_items=32):
    print("--- Starting Deduplication Process ---")
    start_time = time.time()

    print("Grouping collections...")
    col_groups = df.groupby('collection_id')['beatmap_id'].apply(set).to_dict()

    print("Step 1: Removing Exact Duplicates...")
    content_hashes = {}
    for cid, bms in col_groups.items():
        sig = tuple(sorted(bms))
        prev = content_hashes.get(sig)
        if prev is None or cid < prev:
            content_hashes[sig] = cid

    unique_content_cids = set(content_hashes.values())
    print(f"Reduced from {len(col_groups)} to {len(unique_content_cids)} unique content sets.")

    print(f"Step 2: Fuzzy Deduplication (Threshold: {threshold})...")

    sorted_cids = sorted(unique_content_cids, key=lambda x: len(col_groups[x]), reverse=True)

    kept_cids = []
    kept_sets = {}  
    postings = defaultdict(list)  

    def min_required_overlap(len_a, len_b, t):
        return math.ceil((t * (len_a + len_b)) / (1.0 + t))

    for cid in tqdm(sorted_cids, desc="Deduplicating"):
        A = col_groups[cid]
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
            candidates = sorted(overlap_counts.items(), key=lambda kv: kv[1], reverse=True)

            for kept_cid, approx_overlap in candidates:
                B = kept_sets[kept_cid]
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
            kept_cids.append(cid)
            kept_sets[cid] = A
            for bm in A:
                postings[bm].append(cid)

    print(f"Final Collection Count: {len(kept_cids)} (Removed {len(col_groups) - len(kept_cids)} duplicates)")
    print(f"Deduplication took {(time.time() - start_time):.2f}s")

    return df[df['collection_id'].isin(kept_cids)].copy()

def run_nmf():
    print("--- Loading Data ---")
    df = pd.read_parquet(COLLECTIONS_DATA_PATH)

    col_counts = df.groupby('collection_id')['beatmap_id'].count()
    valid_collections = col_counts[
        (col_counts >= MIN_MAPS_IN_COLLECTION) & 
        (col_counts <= MAX_MAPS_IN_COLLECTION)
    ].index
    df = df[df['collection_id'].isin(valid_collections)].copy()
    
    df = deduplicate_collections(df)

    bm_counts = df['beatmap_id'].value_counts()
    max_allowed = len(df['collection_id'].unique()) * MAX_FREQUENCY_PER_MAP
    valid_maps = bm_counts[
        (bm_counts >= MIN_COLLECTIONS_PER_MAP) & 
        (bm_counts <= max_allowed)
    ].index
    df = df[df['beatmap_id'].isin(valid_maps)].copy()

    print("--- Applying Set/Song Dampening ---")
    df['song_occurrence'] = df.groupby(['collection_id', 'beatmap_name'])['beatmap_id'].transform('count')
    df['weight'] = 1.0 / (df['song_occurrence'] ** SONG_DAMPENING_POWER)
    
    unique_collections = sorted(df['collection_id'].unique())
    unique_beatmaps = sorted(df['beatmap_id'].unique())
    
    col_to_idx = {cid: i for i, cid in enumerate(unique_collections)}
    bm_to_idx = {bid: i for i, bid in enumerate(unique_beatmaps)}
    idx_to_bm = {i: bid for bid, i in bm_to_idx.items()}
    bm_names = df.drop_duplicates('beatmap_id').set_index('beatmap_id')['beatmap_name'].to_dict()

    print(f"Matrix: {len(unique_collections)} Col x {len(unique_beatmaps)} Maps")

    row = df['collection_id'].map(col_to_idx).values
    col = df['beatmap_id'].map(bm_to_idx).values
    data = df['weight'].values
    
    X = csr_matrix((data, (row, col)), shape=(len(unique_collections), len(unique_beatmaps)))

    print("Applying TF-IDF (L2 Norm)...")
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    X_tfidf = tfidf.fit_transform(X) *  SCALING_FACTOR

    print(f"Running NMF ({N_TOPICS} topics)...")
    nmf = NMF(
        n_components=N_TOPICS,
        init='nndsvd',       
        beta_loss='frobenius',
        solver='cd',         
        tol=1e-5,            
        max_iter=500,
        random_state=42,
        alpha_W=ALPHA,       
        alpha_H=ALPHA,
        l1_ratio=L1_RATIO,
        verbose=1
    )

    start = time.time()
    W = nmf.fit_transform(X_tfidf)
    H = nmf.components_
    print(f"Converged in {(time.time()-start)/60:.2f} minutes")

    print("Calculating topic statistics...")
    col_counts = pd.Series(W.argmax(axis=1)).value_counts()
    bm_counts = pd.Series(H.argmax(axis=0)).value_counts()
    
    stats_df = pd.DataFrame([{
        'topic_id': i,
        'collections': int(col_counts.get(i, 0)),
        'beatmaps': int(bm_counts.get(i, 0))
    } for i in range(N_TOPICS)])
    
    print("\nTopic Distribution Statistics:")
    print(stats_df.to_string(index=False))

    print("Saving weights as parquet...")
    h_records = []
    for t_idx in range(N_TOPICS):
        relevant_b_indices = np.where(H[t_idx] > 1e-4)[0]
        for b_idx in relevant_b_indices:
            h_records.append({
                'beatmap_id': unique_beatmaps[b_idx],
                'topic_id': t_idx,
                'weight': float(H[t_idx, b_idx])
            })
    pd.DataFrame(h_records).to_parquet(f"beatmap_topic_weights_{VERSION}.parquet")

    w_records = []
    for c_idx in range(len(unique_collections)):
        relevant_t_indices = np.where(W[c_idx] > 1e-4)[0]
        for t_idx in relevant_t_indices:
            w_records.append({
                'collection_id': unique_collections[c_idx],
                'topic_id': t_idx,
                'weight': float(W[c_idx, t_idx])
            })
    pd.DataFrame(w_records).to_parquet(f"collection_topic_weights_{VERSION}.parquet")

    print("Generating summaries...")
    
    map_topic_counts = {}
    for t_idx in range(N_TOPICS):
        top_indices = H[t_idx].argsort()[::-1][:30]
        map_topic_counts.update({idx: map_topic_counts.get(idx, 0) + 1 for idx in top_indices})

    summary = []
    for t_idx in range(N_TOPICS):
        top_indices = H[t_idx].argsort()[::-1][:30]
        names = []
        for i in top_indices:
            is_generic = map_topic_counts.get(i, 0) > (N_TOPICS * 0.10)
            bid = idx_to_bm[i]
            name = bm_names.get(bid, 'Unknown').replace("|", "-").replace(",", " ")
            names.append(f"*{name}*" if is_generic else name)
        
        summary.append({'topic_id': t_idx, 'top_maps': " | ".join(names[:20])})
    summary_df = pd.DataFrame(summary)

    all_top_cids = set()
    top_cids_per_topic = []
    for t_idx in range(N_TOPICS):
        top_indices = W[:, t_idx].argsort()[::-1][:5]
        cids = [unique_collections[i] for i in top_indices]
        top_cids_per_topic.append(cids)
        all_top_cids.update(cids)
        
    col_names_map = get_collection_names(list(all_top_cids))
    col_summary = []
    for t_idx in range(N_TOPICS):
        cids = top_cids_per_topic[t_idx]
        names = [f"{col_names_map.get(cid, 'Unknown')} ({cid})" for cid in cids]
        col_summary.append({'topic_id': t_idx, 'top_collections': " | ".join(names)})
    col_summary_df = pd.DataFrame(col_summary)

    final_summary = stats_df \
        .merge(col_summary_df, on='topic_id', how='left') \
        .merge(summary_df, on='topic_id', how='left')
    
    final_summary.columns = ['topic_id', 'collection_count', 'beatmap_count', 'collection_names', 'beatmap_names']
    final_summary.to_csv(COLLECTIONS_DIR / f"topic_summary_{VERSION}.csv", index=False)
    print(f"Fused summary saved to {COLLECTIONS_DIR / f'topic_summary_{VERSION}.csv'}")
    
    print("Done.")

if __name__ == "__main__":
    run_nmf()
