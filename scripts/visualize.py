# visualize.py
import os
import sys
import argparse
import random
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config
from core.data.loader import load_dataset
from core.data.transforms import BeatmapNormalizer, BeatmapTransform
from core.data.dataset import collate_fn
from core.model.bert import BertForContrastiveFineTuning
from core.training.checkpoint import CheckpointManager

import umap
import plotly.graph_objects as go

# --- Constants ---
DEFAULT_SAMPLE_SIZE = 500
CONFIG_PATH = "./config.yaml"
CHECKPOINT_DIR = "checkpoints"
PRETRAIN_MODEL_NAME = "model"
FINETUNED_MODEL_NAME = "model_finetuned"

# --- Helper Functions (No changes here) ---

def get_beatmap_ids_in_order(dataset_path: str) -> pd.DataFrame:
    print("Reading beatmap IDs from source...")
    beatmaps_df = pd.read_parquet(os.path.join(dataset_path, 'beatmaps'))
    hitobjects_df = pd.read_parquet(os.path.join(dataset_path, 'hitobjects'))
    df = pd.merge(hitobjects_df, beatmaps_df, on='beatmap_id', how='inner')
    map_counts = df['beatmap_id'].value_counts()
    valid_beatmap_ids = map_counts[map_counts >= 2].index
    if len(valid_beatmap_ids) < len(beatmaps_df):
        df = df[df['beatmap_id'].isin(valid_beatmap_ids)].copy()
    df_sorted = df.sort_values(['beatmap_id', 'time'])
    columns_to_select = ['beatmap_id']
    if 'beatmapset_id' in df_sorted.columns:
        columns_to_select.append('beatmapset_id')
    unique_beatmap_info = df_sorted[columns_to_select].drop_duplicates('beatmap_id')
    return unique_beatmap_info

def load_model_and_normalizer(config: dict, device: torch.device, full_dataset: list) -> tuple:
    print("Loading model and normalization stats...")
    model = BertForContrastiveFineTuning.from_config(config, device)
    model.eval()
    pretrain_manager = CheckpointManager(CHECKPOINT_DIR, model_name=PRETRAIN_MODEL_NAME)
    stats = pretrain_manager.load_normalization_stats()
    if stats is None:
        print("\n" + "="*80)
        print("WARNING: Could not load normalization stats from pre-trained checkpoint.")
        print("Calculating new normalization stats from the loaded dataset for this run.")
        print("="*80 + "\n")
        normalizer = BeatmapNormalizer.from_data(full_dataset, include_augmentation=False)
    else:
        vector_stats, meta_stats = stats
        normalizer = BeatmapNormalizer(vector_stats=vector_stats, meta_stats=meta_stats)
        print("Successfully created normalizer from pre-trained stats.")
    finetune_manager = CheckpointManager(CHECKPOINT_DIR, model_name=FINETUNED_MODEL_NAME)
    ckpt_path = finetune_manager.get_checkpoint_path('latest')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    compiled_prefix = '_orig_mod.'
    is_checkpoint_compiled = any(key.startswith(compiled_prefix) for key in state_dict.keys())
    is_model_compiled = any(key.startswith(compiled_prefix) for key in model.state_dict().keys())
    if is_checkpoint_compiled and not is_model_compiled:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[len(compiled_prefix):] if key.startswith(compiled_prefix) else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    elif not is_checkpoint_compiled and is_model_compiled:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[compiled_prefix + key] = value
        state_dict = new_state_dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        optional_heads = ['user_tag_head', 'user_tag_projection']
        critical_missing = [k for k in missing_keys if not any(head in k for head in optional_heads)]
        if critical_missing:
            print(f"ERROR: Critical keys missing from checkpoint: {critical_missing}")
        else:
            print(f"Info: Ignored missing optional keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint were ignored: {unexpected_keys}")
    print(f"Loaded fine-tuned model weights from {ckpt_path}")
    return model, normalizer

class InferenceDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        vectors, metadata = self.data[idx]
        return self.transform(vectors, metadata)

@torch.no_grad()
def generate_embeddings(model, normalizer, data_subset, config, device) -> np.ndarray:
    print(f"Generating embeddings for {len(data_subset)} beatmaps...")
    transform = BeatmapTransform(normalizer, augment=False)
    dataset = InferenceDataset(data_subset, transform)
    actual_vector_dim = data_subset[0][0].shape[1]
    collate_with_args = lambda batch: collate_fn(
        batch, max_seq_len=config['data']['max_seq_len'], vector_dim=actual_vector_dim, device=device
    )
    dataloader = DataLoader(
        dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_with_args
    )
    all_embeddings = []
    for batch in tqdm(dataloader, desc="Generating Embeddings"):
        vectors, attention_mask, metadata = batch
        predictions = model(vectors, metadata, attention_mask)
        all_embeddings.append(predictions['cls_representation'].cpu())
    return torch.cat(all_embeddings, dim=0).float().numpy()


def main():
    parser = argparse.ArgumentParser(description="Generate and visualize beatmap embeddings.")
    parser.add_argument("--test", action="store_true", help="Use the test dataset.")
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of beatmaps to sample.")
    args = parser.parse_args()
    
    dataset_name = "beatmap_dataset_test" if args.test else "beatmap_dataset"
    dataset_path = os.path.join("data", dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Using dataset: {dataset_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = load_config("config", config_dir=".")

    all_data, all_ratings = load_dataset(dataset_path, config['data']['max_seq_len'])
    all_ids_df = get_beatmap_ids_in_order(dataset_path)

    model, normalizer = load_model_and_normalizer(config, device, all_data)

    num_beatmaps = min(len(all_data), len(all_ids_df))
    sample_size = min(args.sample_size, num_beatmaps)

    print(f"Sampling {sample_size} beatmaps from a total of {num_beatmaps}...")
    indices = random.sample(range(num_beatmaps), sample_size)
    
    sampled_data = [all_data[i] for i in indices]
    sampled_ratings = all_ratings[indices]
    sampled_ids_df = all_ids_df.iloc[indices]
 
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        embeddings = generate_embeddings(model, normalizer, sampled_data, config, device)

    print("Performing dimensionality reduction with UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings)

    print("Creating interactive visualization with Plotly...")
    
    has_set_id = 'beatmapset_id' in sampled_ids_df.columns

    if has_set_id:
        print("Found beatmapset_id, will generate links to beatmap sets (/s/).")
        custom_data = np.stack((
            sampled_ids_df['beatmapset_id'].values,
            sampled_ids_df['beatmap_id'].values
        ), axis=-1)
        hovertemplate = (
            "<b>Set ID:</b> %{customdata[0]}<br>"
            "<b>Map ID:</b> %{customdata[1]}<br>"
            "<b>Difficulty:</b> %{marker.color:.2f}<br>"
            "<b>Click to open beatmap set</b>"
            "<extra></extra>"
        )
    else:
        print("Warning: beatmapset_id not found. Generating links to individual beatmaps (/b/).")
        custom_data = sampled_ids_df['beatmap_id'].values
        hovertemplate = (
            "<b>Map ID:</b> %{customdata}<br>"
            "<b>Difficulty:</b> %{marker.color:.2f}<br>"
            "<b>Click to open beatmap</b>"
            "<extra></extra>"
        )

    fig = go.Figure(data=go.Scatter(
        x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], mode='markers',
        marker=dict(
            color=sampled_ratings, colorscale='Viridis', showscale=True,
            colorbar_title_text='Difficulty', size=8, opacity=0.8,
        ),
        customdata=custom_data,
        hovertemplate=hovertemplate
    ))

    fig.update_layout(
        title=f'2D UMAP Visualization of {sample_size} Beatmap Embeddings',
        xaxis_title='UMAP Dimension 1', yaxis_title='UMAP Dimension 2',
        xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
        hovermode='closest'
    )

    # --- START: FINAL FIX FOR CLICKABLE LINKS ---
    
    # This raw JavaScript code is passed to Plotly, which will wrap it in <script> tags correctly.
    # The <script> tags are NOT included in the string itself.
    js_script = """
    window.addEventListener('load', function() {
        var plot_div = document.querySelector('.plotly-graph-div');
        if (plot_div) {
            console.log("Plotly graph div found. Adding click listener.");
            plot_div.on('plotly_click', function(data){
                if(data.points.length > 0) {
                    var point = data.points[0];
                    var customData = point.customdata;
                    var url;

                    if (Array.isArray(customData) && customData.length > 0) {
                        var setId = customData[0];
                        url = 'https://osu.ppy.sh/s/' + setId;
                    } else if (customData) {
                        var mapId = customData;
                        url = 'https://osu.ppy.sh/b/' + mapId;
                    }

                    if (url) {
                        console.log('Opening URL: ' + url);
                        window.open(url, '_blank');
                    } else {
                        console.log('No valid URL could be constructed from custom data:', customData);
                    }
                }
            });
        } else {
            console.error("Could not find Plotly graph div to attach click event.");
        }
    });
    """

    output_file = 'embeddings_visualization.html'
    
    fig.write_html(output_file, post_script=js_script, include_plotlyjs='cdn')
    
    print(f"\n--- Visualization Complete! ---")
    print(f"Interactive graph saved to: {os.path.abspath(output_file)}")
    print("You can now open this file in a browser. Clicking on a point will open the beatmap link.")
    # --- END: FINAL FIX FOR CLICKABLE LINKS ---


if __name__ == "__main__":
    main()