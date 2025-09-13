import os
import sqlite3
import sys
import itertools
import gdown
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional

def setup_database(db_path: str, colab_url: Optional[str] = None) -> str:
    try:
        import google.colab  # type: ignore
        if not os.path.exists('/content/beatmaps.db') and colab_url:
            print("Downloading database for Colab environment...")
            gdown.download(colab_url, '/content/beatmaps.db', quiet=False)
        return '/content/beatmaps.db'
    except ImportError:
        return db_path


def load_and_group_data_from_db(
    db_path: str, 
    chunk_size: int = 1000,
    max_seq_len: Optional[int] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    print("Connecting to database...")
    con = sqlite3.connect(db_path)
    cursor = con.cursor()

    print("Fetching valid beatmap IDs and all metadata...")
    metadata_df = pd.read_sql_query(
        """
        SELECT id, ar, od, circle_size as cs, difficulty_rating, main_bpm 
        FROM beatmaps 
        WHERE main_bpm IS NOT NULL 
        ORDER BY id
        """, 
        con
    )
    valid_map_ids = metadata_df['id'].tolist()
    
    metadata_dict = {
        row.id: np.array([row.ar, row.od, row.cs, row.difficulty_rating, row.main_bpm], dtype=np.float32)
        for row in metadata_df.itertuples(index=False)
    }
    
    print(f"Found {len(valid_map_ids)} beatmaps with complete metadata.")

    processed_data = []
    num_chunks = (len(valid_map_ids) + chunk_size - 1) // chunk_size

    vector_query_template = """
        SELECT beatmap_id, x_diff, y_diff, time_diff, abs_x, abs_y, object_type, is_new_combo, slider_curve_type, slider_num_anchors, slider_pixel_length
        FROM beatmap_vectors 
        WHERE beatmap_id IN ({placeholders}) 
        ORDER BY beatmap_id
    """

    for i in tqdm(
        range(0, len(valid_map_ids), chunk_size),
        total=num_chunks,
        desc="Processing Chunks",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout
    ):
        chunk_ids = valid_map_ids[i:i + chunk_size]
        
        placeholders = ','.join('?' for _ in chunk_ids)
        query = vector_query_template.format(placeholders=placeholders)
        
        cursor.execute(query, chunk_ids)
        
        for map_pk, group_iter in itertools.groupby(cursor, key=lambda row: row[0]):
            vectors_list = [row[1:] for row in group_iter]
            
            if not vectors_list:
                continue

            meta_np = metadata_dict.get(map_pk)
            if meta_np is None:
                continue 

            vectors_tensor = torch.tensor(vectors_list, dtype=torch.float32)
            vectors_tensor[:, 2] = torch.log1p(vectors_tensor[:, 2])  # Log transform time_diff
            
            if max_seq_len and vectors_tensor.shape[0] > max_seq_len:
                vectors_tensor = vectors_tensor[:max_seq_len]
            
            metadata_tensor = torch.from_numpy(meta_np) 
            processed_data.append((vectors_tensor, metadata_tensor))

    con.close()
    print("Finished processing all data.")
    return processed_data


def calculate_normalization_stats(
    train_data: List[Tuple[torch.Tensor, torch.Tensor]],
    include_augmentation: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    print("Calculating normalization statistics...")
    
    all_vectors_list = [data[0] for data in train_data]
    all_metadata_list = [data[1] for data in train_data]

    if include_augmentation:
        print("Including data augmentation in normalization statistics...")
        augmented_vectors_list = []
        for vectors in all_vectors_list:
            augmented_vectors_list.append(vectors)
            
            flipped_x = vectors.clone()
            flipped_x[:, 0] *= -1                  
            flipped_x[:, 3] = 512 - flipped_x[:, 3]  
            augmented_vectors_list.append(flipped_x)
            
            flipped_y = vectors.clone()
            flipped_y[:, 1] *= -1                  
            flipped_y[:, 4] = 384 - flipped_y[:, 4]  
            augmented_vectors_list.append(flipped_y)

            flipped_xy = vectors.clone()
            flipped_xy[:, 0] *= -1                  
            flipped_xy[:, 1] *= -1                  
            flipped_xy[:, 3] = 512 - flipped_xy[:, 3]  
            flipped_xy[:, 4] = 384 - flipped_xy[:, 4]  
            augmented_vectors_list.append(flipped_xy)
            
        all_vectors_tensor = torch.cat(augmented_vectors_list, dim=0)
    else:
        all_vectors_tensor = torch.cat(all_vectors_list, dim=0)
    
    all_metadata_tensor = torch.stack(all_metadata_list, dim=0) 

    vector_mean = all_vectors_tensor.mean(dim=0)
    vector_std = all_vectors_tensor.std(dim=0)
    meta_mean = all_metadata_tensor.mean(dim=0)
    meta_std = all_metadata_tensor.std(dim=0)

    vector_std[vector_std == 0] = 1.0
    meta_std[meta_std == 0] = 1.0

    print("--- Normalization Statistics ---")
    print(f"Vector Mean: {vector_mean.numpy()}")
    print(f"Vector Std:  {vector_std.numpy()}")
    print(f"Meta Mean:   {meta_mean.numpy()}")
    print(f"Meta Std:    {meta_std.numpy()}")
    
    return vector_mean, vector_std, meta_mean, meta_std


def create_weighted_sampler(
    data: List[Tuple[torch.Tensor, torch.Tensor]],
    difficulty_index: int = 3
) -> WeightedRandomSampler:
    print("Creating weighted sampler for difficulty balancing...")
    
    all_difficulty_ratings = np.array([item[1][difficulty_index].item() for item in data])
    
    bins = [0, 5, 6, 7, 8, 9, 10, np.inf]
    binned_ratings = pd.cut(all_difficulty_ratings, bins=bins, right=False, labels=False)
    class_counts = np.bincount(binned_ratings, minlength=len(bins)-1)

    print("\n--- Difficulty Distribution ---")
    for i in range(len(bins)-1):
        lower = bins[i]
        upper = bins[i+1]
        count = class_counts[i]
        total_maps = len(data)
        percentage = (count / total_maps * 100) if total_maps > 0 else 0
        if np.isinf(upper):
            print(f"{lower:2.0f}★+  : {count:7d} maps ({percentage:5.2f}%)")
        else:
            print(f"{lower:2.0f}-{upper:2.0f}★ : {count:7d} maps ({percentage:5.2f}%)")
    print("-" * 35)

    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[binned_ratings]
    sample_weights = torch.from_numpy(sample_weights).double()
    
    return WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )


def print_data_summary(all_data: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Prints a summary of the loaded data."""
    if not all_data:
        print("No data loaded!")
        return
        
    total_maps = len(all_data)
    metadata_dim = all_data[0][1].shape[0]
    vector_dim = all_data[0][0].shape[1]
    
    seq_lengths = [data[0].shape[0] for data in all_data]
    avg_seq_len = np.mean(seq_lengths)
    max_seq_len = np.max(seq_lengths)
    min_seq_len = np.min(seq_lengths)
    
    print(f"\n--- Data Summary ---")
    print(f"Total beatmaps: {total_maps}")
    print(f"Vector dimension: {vector_dim}")
    print(f"Metadata dimension: {metadata_dim}")
    print(f"Sequence length - Min: {min_seq_len}, Max: {max_seq_len}, Avg: {avg_seq_len:.1f}")
    print("-" * 20)