# visualize.py
import os
import sys
import argparse
import random
import json
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

DEFAULT_SAMPLE_SIZE = 500
CONFIG_PATH = "./config.yaml"
CHECKPOINT_DIR = "checkpoints"
PRETRAIN_MODEL_NAME = "model"
FINETUNED_MODEL_NAME = "model_finetuned"
LABELS_PATH = "./data/labels.json"

def get_beatmap_ids_in_order(dataset_path: str) -> pd.DataFrame:
    print("Reading beatmap IDs from source (memory-efficient)...")
    beatmaps_path = os.path.join(dataset_path, 'beatmaps')
    hitobjects_path = os.path.join(dataset_path, 'hitobjects')

    if not os.path.exists(beatmaps_path) or not os.path.exists(hitobjects_path):
        raise FileNotFoundError(f"Required Parquet files not found in {dataset_path}")

    ho_ids = pd.read_parquet(hitobjects_path, columns=['beatmap_id'])
    
    map_counts = ho_ids['beatmap_id'].value_counts()
    valid_beatmap_ids = map_counts[map_counts >= 2].index

    beatmaps_df = pd.read_parquet(beatmaps_path)
    
    final_beatmaps_df = beatmaps_df[beatmaps_df['beatmap_id'].isin(valid_beatmap_ids)].copy()

    initial_count = len(final_beatmaps_df)
    final_beatmaps_df.drop_duplicates(subset=['beatmap_id'], keep='first', inplace=True)
    final_count = len(final_beatmaps_df)
    if initial_count != final_count:
        print(f"Warning: Removed {initial_count - final_count} duplicate beatmap IDs from the source list.")

    columns_to_select = ['beatmap_id']
        
    return final_beatmaps_df[columns_to_select].reset_index(drop=True)

def load_model_and_normalizer(config: dict, device: torch.device) -> tuple:
    print("Loading model and normalization stats...")
    model = BertForContrastiveFineTuning.from_config(config, device)
    model.eval()
    pretrain_manager = CheckpointManager(CHECKPOINT_DIR, model_name=PRETRAIN_MODEL_NAME)
    
    print("Attempting to load normalization stats from pre-trained checkpoint...")
    vector_stats = pretrain_manager.load_normalization_stats()

    if vector_stats is None:
        raise FileNotFoundError(
            f"Could not load normalization stats from pre-trained checkpoint. "
            f"Ensure a checkpoint for '{PRETRAIN_MODEL_NAME}' exists in the '{CHECKPOINT_DIR}' "
            "directory and contains the necessary normalization stats."
        )

    normalizer = BeatmapNormalizer(vector_stats=vector_stats)
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
        vectors = self.data[idx]
        return self.transform(vectors)

@torch.no_grad()
def generate_embeddings(model, normalizer, data_subset, config, device) -> np.ndarray:
    print(f"Generating embeddings for {len(data_subset)} beatmaps...")
    transform = BeatmapTransform(normalizer, augment=False)
    dataset = InferenceDataset(data_subset, transform)
    actual_vector_dim = data_subset[0].shape[1]
    collate_with_args = lambda batch: collate_fn(
        batch, max_seq_len=config['data']['max_seq_len'], vector_dim=actual_vector_dim, device=device
    )
    dataloader = DataLoader(
        dataset, batch_size=config['pretraining']['batch_size'], shuffle=False, collate_fn=collate_with_args
    )
    all_embeddings = []
    for batch in tqdm(dataloader, desc="Generating Embeddings"):
        vectors, attention_mask = batch
        predictions = model(vectors, attention_mask)
        all_embeddings.append(predictions['sequence_representation'].cpu())
    return torch.cat(all_embeddings, dim=0).float().numpy()


def main():
    parser = argparse.ArgumentParser(description="Generate and visualize beatmap embeddings.")
    parser.add_argument("--test", action="store_true", help="Use the test dataset.")
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of beatmaps to sample.")
    parser.add_argument("--labelled", action="store_true", help="Visualize only labeled beatmaps.")
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

    labels_dict = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r') as f:
            labels_dict = json.load(f)

    all_ids_df = get_beatmap_ids_in_order(dataset_path)

    if args.labelled:
        if not labels_dict:
            print(f"Error: --labelled flag was used, but no labels found in {LABELS_PATH}", file=sys.stderr)
            sys.exit(1)
        
        print("Filtering for labeled beatmaps only...")
        labeled_ids_with_labels = {int(bid) for bid, labels in labels_dict.items() if labels}
        if not labeled_ids_with_labels:
            print(f"Error: Labels file {LABELS_PATH} exists, but contains no beatmaps with labels.", file=sys.stderr)
            sys.exit(1)
        
        all_ids_df = all_ids_df[all_ids_df['beatmap_id'].isin(labeled_ids_with_labels)]
        print(f"Found {len(all_ids_df)} labeled beatmaps in the dataset to sample from.")

    else:
        if labels_dict:
            print("Excluding labeled beatmaps from visualization...")
            labeled_ids = {int(bid) for bid, labels in labels_dict.items() if labels}
            initial_count = len(all_ids_df)
            all_ids_df = all_ids_df[~all_ids_df['beatmap_id'].isin(labeled_ids)]
            num_excluded = initial_count - len(all_ids_df)
            print(f"Excluded {num_excluded} labeled beatmaps from the sampling pool.")
        else:
            print(f"Warning: No labels file found at {LABELS_PATH}. Visualizing from all available beatmaps.")
    
    num_beatmaps = len(all_ids_df)

    if num_beatmaps == 0:
        if args.labelled:
            print("Error: No labeled beatmaps that exist in the dataset could be found.", file=sys.stderr)
        else:
            print("Error: No valid, unlabeled beatmaps found in the dataset.", file=sys.stderr)
        sys.exit(1)

    sample_size = min(args.sample_size, num_beatmaps)
    print(f"Sampling {sample_size} beatmaps from a total of {num_beatmaps} candidates...")
    sampled_ids_df = all_ids_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    ids_to_load = sampled_ids_df['beatmap_id'].tolist()

    sampled_data, difficulty_attrs, loaded_ids = load_dataset(
        dataset_path, 
        config['data']['max_seq_len'],
        ids_to_load=ids_to_load
    )

    if len(loaded_ids) != len(ids_to_load):
        print(f"Warning: Requested {len(ids_to_load)} maps, but loaded {len(loaded_ids)}. "
              "This may be due to filtering or missing data for some IDs.")

    sampled_ratings = difficulty_attrs.get('stars') if isinstance(difficulty_attrs, dict) else None
    if sampled_ratings is None:
        sampled_ratings = np.zeros(len(loaded_ids), dtype=np.float32)
    else:
        sampled_ratings = np.asarray(sampled_ratings, dtype=np.float32)
        if sampled_ratings.shape[0] != len(loaded_ids):
            print(f"Warning: Expected {len(loaded_ids)} difficulty ratings but received "
                  f"{sampled_ratings.shape[0]}. Truncating to match embeddings.", file=sys.stderr)
            sampled_ratings = sampled_ratings[:len(loaded_ids)]

    model, normalizer = load_model_and_normalizer(config, device)
 
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        embeddings = generate_embeddings(model, normalizer, sampled_data, config, device)

    print("Performing dimensionality reduction with UMAP...")
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42, 
        n_neighbors=5, 
        min_dist=0.1
        )
    embeddings_2d = reducer.fit_transform(embeddings)

    print("Creating interactive visualization with Plotly...")
    
    if not (len(embeddings_2d) == len(sampled_ratings) == len(loaded_ids)):
        print("ERROR: Mismatch in lengths of data for plotting. Aborting.", file=sys.stderr)
        print(f"Embeddings: {len(embeddings_2d)}, Ratings: {len(sampled_ratings)}, Loaded IDs: {len(loaded_ids)}", file=sys.stderr)
        sys.exit(1)

    if args.labelled:
        labels_list = [', '.join(labels_dict.get(str(bid), [])) for bid in loaded_ids]
        custom_data = list(zip(loaded_ids, labels_list))
        hovertemplate = (
            "<b>Map ID:</b> %{customdata[0]}<br>"
            "<b>Labels:</b> %{customdata[1]}<br>"
            "<b>Difficulty:</b> %{marker.color:.2f}<br>"
            "<b>Click to open beatmap</b>"
            "<extra></extra>"
        )
        title_suffix = 'Labeled Beatmap Embeddings'
    else:
        custom_data = loaded_ids
        hovertemplate = (
            "<b>Map ID:</b> %{customdata}<br>"
            "<b>Difficulty:</b> %{marker.color:.2f}<br>"
            "<b>Click to open beatmap</b>"
            "<extra></extra>"
        )
        title_suffix = 'Unlabeled Beatmap Embeddings'

    fig = go.Figure(data=go.Scattergl(
        x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], mode='markers',
        marker=dict(
            color=sampled_ratings, colorscale='Viridis', showscale=True,
            colorbar_title_text='Difficulty', size=8, opacity=0.8,
        ),
        customdata=custom_data,
        hovertemplate=hovertemplate
    ))

    fig.update_layout(
        title=f'2D UMAP Visualization of {sample_size} {title_suffix}',
        xaxis_title='UMAP Dimension 1', yaxis_title='UMAP Dimension 2',
        xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
        hovermode='closest'
    )

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
                    var mapId;

                    if (Array.isArray(customData) && customData.length > 0) {
                        // For labeled case: customData is [mapId, labels]
                        mapId = customData[0];
                    } else if (customData) {
                        // For unlabeled case: customData is just mapId
                        mapId = customData;
                    }

                    if (mapId) {
                        url = 'https://osu.ppy.sh/b/' + mapId;
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

    output_file = './html/embeddings_visualization.html'
    
    fig.write_html(output_file, post_script=js_script, include_plotlyjs='cdn')
    
    print(f"\n--- Visualization Complete! ---")
    print(f"Interactive graph saved to: {os.path.abspath(output_file)}")
    print("You can now open this file in a browser. Clicking on a point will open the beatmap link.")


if __name__ == "__main__":
    main()