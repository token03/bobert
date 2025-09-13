import os
import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from torch.utils.data import random_split
import itertools

# --- New Imports for Visualization ---
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import webbrowser

# ------------------------------------

import logging
logging.getLogger('chromadb').setLevel(logging.ERROR)


print("--- Initializing Setup and Constants ---")

DB_PATH = './beatmaps.db'
CHECKPOINT_PATH = "./checkpoints/mlm_bert_rope_latest.pth"
OUTPUT_HTML_PATH = "./beatmap_similarity_graph_sampled.html" # Changed output file name

# --- New Constants for Sampling ---
SAMPLE_SIZE_PER_RANGE = 2000
SR_RANGES_TO_SAMPLE = [
    (5.0, 6.0), # 5-star maps (>= 5.0 and < 6.0)
    (6.0, 7.0), # 6-star maps
    (7.0, 8.0), # 7-star maps
]
# ------------------------------------

MAX_SEQ_LEN = 1023
IN_CHANNELS = 11  # Updated from 10 to include duration_beats
METADATA_DIM = 5  # ar, od, cs, difficulty_rating, bpm

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
DIM_FEEDFORWARD = 4 * D_MODEL
DROPOUT = 0.1
BATCH_SIZE = 128 # Used for batch-embedding

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ===================================================================
# 2. MODEL DEFINITION (OsuBert with RoPE)
# ===================================================================
print("--- Defining OsuBert Model Architecture with RoPE ---")

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes the complex numbers for RoPE rotations."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """Applies RoPE to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, freqs_cis, mask):
        batch_size, seq_len, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = mask.unsqueeze(1).unsqueeze(2)
        attn_mask = attn_mask == False

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.wo(output)

class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, dropout)
        self.w1 = nn.Linear(d_model, dim_feedforward)
        self.w2 = nn.Linear(dim_feedforward, d_model)
        self.w3 = nn.Linear(d_model, dim_feedforward)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, freqs_cis, src_key_padding_mask):
        src2 = self.self_attn(self.norm1(src), freqs_cis, src_key_padding_mask)
        src = src + self.dropout1(src2)

        normalized_src = self.norm2(src)
        ffn_output = self.w2(F.silu(self.w1(normalized_src)) * self.w3(normalized_src))
        src = src + self.dropout2(ffn_output)

        return src

class OsuBert(nn.Module):
    def __init__(self, *, max_seq_len, d_model, n_heads, n_layers, dim_feedforward, dropout, metadata_dim, in_channels):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(in_channels, d_model)
        self.metadata_proj = nn.Linear(metadata_dim, d_model)
        self.metadata_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model // n_heads, max_seq_len + 1))

    def encode(self, embeddings, padding_mask):
        """Runs the transformer encoder layers on already-embedded inputs."""
        freqs_cis_slice = self.freqs_cis[:embeddings.shape[1]]

        output = embeddings
        for layer in self.layers:
            output = layer(output, freqs_cis=freqs_cis_slice, src_key_padding_mask=padding_mask)
        return output

    def forward(self, x, metadata, attention_mask):
        """Performs full forward pass from raw inputs to final embeddings."""
        batch_size, _, _ = x.shape

        # --- Embedding Stage ---
        x_embed = self.input_proj(x)
        meta_embed = self.metadata_proj(metadata).unsqueeze(1) + self.metadata_token
        full_embeddings = torch.cat([meta_embed, x_embed], dim=1)

        # --- Mask Creation ---
        meta_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        padding_mask = ~attention_mask
        full_padding_mask = torch.cat([meta_mask, padding_mask], dim=1)

        # --- Encoding Stage ---
        output = self.encode(full_embeddings, full_padding_mask)

        return output


# ===================================================================
# 3. HELPER FUNCTIONS FOR DATA, MODEL, AND EMBEDDING
# ===================================================================

# MODIFIED: This function now performs stratified sampling
def load_sampled_data_from_db(db_path, sr_ranges, sample_size_per_range, chunk_size=1000):
    """
    Loads a stratified random sample of beatmap data from the SQLite database.
    It samples a specified number of maps from each provided star rating range.
    """
    print("Connecting to database and loading a sampled set of beatmap data...")
    con = sqlite3.connect(db_path)
    cursor = con.cursor()

    # Build the stratified sampling query dynamically
    individual_queries = []
    for min_sr, max_sr in sr_ranges:
        query_part = f"""
        SELECT * FROM (
            SELECT id, beatmap_id, ar, od, circle_size as cs, difficulty_rating, main_bpm
            FROM beatmaps
            WHERE main_bpm IS NOT NULL
            AND difficulty_rating IS NOT NULL
            AND difficulty_rating >= {min_sr} AND difficulty_rating < {max_sr}
            ORDER BY RANDOM()
            LIMIT {sample_size_per_range}
        )
        """
        individual_queries.append(query_part)

    metadata_query = "\nUNION ALL\n".join(individual_queries)
    metadata_query = f"SELECT * FROM (\n{metadata_query}\n) ORDER BY id"

    print("Fetching sampled beatmap metadata...")
    metadata_df = pd.read_sql_query(metadata_query, con)

    # --- The rest of the function remains the same ---
    valid_map_pks = metadata_df['id'].tolist()
    pk_to_beatmap_id_map = dict(zip(metadata_df.id, metadata_df.beatmap_id))
    pk_to_sr_map = dict(zip(metadata_df.id, metadata_df.difficulty_rating))
    metadata_dict = {
        row.id: np.array([row.ar, row.od, row.cs, row.difficulty_rating, row.main_bpm], dtype=np.float32)
        for row in metadata_df.itertuples(index=False)
    }

    print(f"Found {len(valid_map_pks)} beatmaps after sampling.")
    if len(valid_map_pks) == 0:
        print("Warning: No beatmaps found for the specified SR ranges. The database might be empty or not contain maps in these ranges.")
        con.close()
        return [], {}, {}


    processed_data = []
    vector_query_template = """
        SELECT beatmap_id, x_diff, y_diff, time_diff, abs_x, abs_y, object_type, is_new_combo, slider_curve_type, slider_num_anchors, slider_pixel_length, duration_beats
        FROM beatmap_vectors
        WHERE beatmap_id IN ({placeholders})
        ORDER BY beatmap_id
    """

    for i in tqdm(
        range(0, len(valid_map_pks), chunk_size),
        desc="Processing Chunks",
        dynamic_ncols=True
    ):
        chunk_ids = valid_map_pks[i:i + chunk_size]
        if not chunk_ids: continue

        placeholders = ','.join('?' for _ in chunk_ids)
        query = vector_query_template.format(placeholders=placeholders)

        cursor.execute(query, chunk_ids)

        for map_pk, group_iter in itertools.groupby(cursor, key=lambda row: row[0]):
            vectors_list = [row[1:] for row in group_iter]
            if not vectors_list: continue

            meta_np = metadata_dict.get(map_pk)
            if meta_np is None: continue

            vectors_tensor = torch.tensor(vectors_list, dtype=torch.float32)
            vectors_tensor[:, 2] = torch.log1p(vectors_tensor[:, 2]) # Apply log transform to time_diff
            metadata_tensor = torch.from_numpy(meta_np)
            processed_data.append((map_pk, vectors_tensor, metadata_tensor))

    con.close()
    print(f"Finished loading and processing data for {len(processed_data)} beatmaps.")
    return processed_data, pk_to_beatmap_id_map, pk_to_sr_map


def calculate_normalization_stats(beatmap_data):
    """Replicates the exact normalization stat calculation from the training script."""
    print("Calculating normalization statistics from the sampled dataset...")
    # Note: Stats will now be based on the sample, not the full dataset.
    # This is generally fine as the sample is diverse enough.
    if len(beatmap_data) < 2:
        print("ERROR: Not enough data to calculate normalization stats. Exiting.")
        sys.exit(1)

    torch.manual_seed(42) # Ensure same split for reproducibility
    val_size = int(len(beatmap_data) * 0.1) if len(beatmap_data) > 10 else 1
    train_size = len(beatmap_data) - val_size
    train_data, _ = random_split(beatmap_data, [train_size, val_size])

    # Data format: (pk, vectors, metadata)
    all_vectors_list = [data[1] for data in train_data]
    all_metadata_list = [data[2] for data in train_data]

    augmented_vectors_list_for_stats = []
    for vectors in all_vectors_list:
        augmented_vectors_list_for_stats.append(vectors)
        # Horizontal flip (X-axis)
        flipped_x = vectors.clone(); flipped_x[:, 0] *= -1; flipped_x[:, 3] = 512 - flipped_x[:, 3]
        augmented_vectors_list_for_stats.append(flipped_x)
        # Vertical flip (Y-axis)
        flipped_y = vectors.clone(); flipped_y[:, 1] *= -1; flipped_y[:, 4] = 384 - flipped_y[:, 4]
        augmented_vectors_list_for_stats.append(flipped_y)
        # Both flips (XY-axis)
        flipped_xy = vectors.clone(); flipped_xy[:, 0] *= -1; flipped_xy[:, 1] *= -1; flipped_xy[:, 3] = 512 - flipped_xy[:, 3]; flipped_xy[:, 4] = 384 - flipped_xy[:, 4]
        augmented_vectors_list_for_stats.append(flipped_xy)

    all_vectors_tensor = torch.cat(augmented_vectors_list_for_stats, dim=0)
    all_metadata_tensor = torch.stack(all_metadata_list, dim=0)

    vector_mean, vector_std = all_vectors_tensor.mean(dim=0), all_vectors_tensor.std(dim=0)
    meta_mean, meta_std = all_metadata_tensor.mean(dim=0), all_metadata_tensor.std(dim=0)

    vector_std[vector_std == 0] = 1.0
    meta_std[meta_std == 0] = 1.0

    print("Normalization stats calculated successfully.")
    return vector_mean, vector_std, meta_mean, meta_std

def load_encoder_from_checkpoint(checkpoint_path):
    """Loads the OsuBert encoder from a saved MLM model checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}"); sys.exit(1)
    print(f"--- Loading Encoder from Checkpoint: {checkpoint_path} ---")
    encoder = OsuBert(
        max_seq_len=MAX_SEQ_LEN, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT, in_channels=IN_CHANNELS, metadata_dim=METADATA_DIM
    )
    full_checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = full_checkpoint.get('model_state_dict', full_checkpoint)
    # Strip the 'bert.' prefix added by the MLM training wrapper
    state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items() if k.startswith("bert.")}
    encoder.load_state_dict(state_dict, strict=True)
    encoder.to(device); encoder.eval()
    print("Encoder model loaded successfully and set to evaluation mode.")
    return encoder

def collate_for_embedding(batch, max_seq_len, vector_dim):
    """Prepare a batch of variable-length sequences for embedding."""
    vectors, metadata = zip(*batch)
    padded = torch.zeros(len(batch), max_seq_len, vector_dim, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)
    for i, v in enumerate(vectors):
        v = v[:max_seq_len]
        length = v.shape[0]
        padded[i, :length] = v
        attention_mask[i, :length] = True
    stacked_metadata = torch.stack(metadata, dim=0)
    return padded.to(device), attention_mask.to(device), stacked_metadata.to(device)


# ===================================================================
# 4. NEW EMBEDDING AND VISUALIZATION FUNCTIONS
# ===================================================================

def generate_all_embeddings(encoder, all_beatmaps, norms):
    """Generates embeddings for all beatmaps in the dataset."""
    print("Generating embeddings for all beatmaps. This may take a while...")
    vector_mean, vector_std, meta_mean, meta_std = norms; epsilon = 1e-8

    all_pks = []
    all_embeddings = []

    num_batches = (len(all_beatmaps) + BATCH_SIZE - 1) // BATCH_SIZE
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Embedding Batches"):
            batch_data = all_beatmaps[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            pks, raw_vectors_list, raw_meta_list = zip(*batch_data)
            norm_vectors = [(v - vector_mean) / (vector_std + epsilon) for v in raw_vectors_list]
            norm_meta = [(m - meta_mean) / (meta_std + epsilon) for m in raw_meta_list]
            padded_x, attention_mask, metadata_tensor = collate_for_embedding(list(zip(norm_vectors, norm_meta)), MAX_SEQ_LEN, IN_CHANNELS)

            encoder_output = encoder(padded_x, metadata_tensor, attention_mask)
            # Use the [CLS] token embedding (the first one, which is our metadata token)
            embeddings = encoder_output[:, 0, :].cpu().numpy()

            all_pks.extend(pks)
            all_embeddings.append(embeddings)

    return all_pks, np.concatenate(all_embeddings, axis=0)

def create_interactive_graph(reduced_embeddings, pks, pk_to_beatmap_id_map, pk_to_sr_map, output_path):
    """Creates and saves an interactive Plotly graph."""
    print(f"Creating interactive graph at {output_path}...")

    # Prepare data for plotting
    x_coords = reduced_embeddings[:, 0]
    y_coords = reduced_embeddings[:, 1]

    beatmap_ids = [pk_to_beatmap_id_map[pk] for pk in pks]
    star_ratings = [pk_to_sr_map[pk] for pk in pks]

    hover_texts = [f"Beatmap ID: {bid}<br>Star Rating: {sr:.2f}" for bid, sr in zip(beatmap_ids, star_ratings)]
    beatmap_urls = [f"https://osu.ppy.sh/b/{bid}" for bid in beatmap_ids]

    # Create the scatter plot
    fig = go.Figure(data=go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(
            size=6,
            color=star_ratings,
            colorscale='Viridis', # A nice color scale
            showscale=True,
            colorbar=dict(
                title="Star Rating"
            ),
            opacity=0.8
        ),
        text=hover_texts,
        hoverinfo='text',
        customdata=beatmap_urls # Store URLs here for the click event
    ))

    # Update layout for a better look
    fig.update_layout(
        title=f'2D Visualization of osu! Beatmap Similarity (Sampled: {len(pks)} maps)',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        template='plotly_dark'
    )

    # Save to HTML
    fig.write_html(output_path, include_plotlyjs='cdn')

    # Add JavaScript for click-to-open-URL functionality
    with open(output_path, 'a') as f:
        f.write("""
        <script>
        var plot = document.getElementsByClassName('plotly-graph-div')[0];
        plot.on('plotly_click', function(data){
            if(data.points.length > 0) {
                var url = data.points[0].customdata;
                window.open(url, '_blank');
            }
        });
        </script>
        """)

    print("Graph created successfully.")
    return output_path

# ===================================================================
# 5. MAIN EXECUTION BLOCK
# ===================================================================
if __name__ == "__main__":
    # 1. Load a stratified random sample from the database
    # MODIFIED: Call the new sampling function
    sampled_beatmaps_data, pk_to_beatmap_id_map, pk_to_sr_map = load_sampled_data_from_db(
        DB_PATH,
        sr_ranges=SR_RANGES_TO_SAMPLE,
        sample_size_per_range=SAMPLE_SIZE_PER_RANGE
    )

    if not sampled_beatmaps_data:
        print("Exiting program as no data was loaded.")
        sys.exit(0)

    # 2. Calculate normalization stats based on the loaded sample
    norms = calculate_normalization_stats(sampled_beatmaps_data)

    # 3. Load the pre-trained encoder model
    encoder_model = load_encoder_from_checkpoint(CHECKPOINT_PATH)

    # 4. Generate embeddings for the sampled beatmaps
    pks, embeddings = generate_all_embeddings(encoder_model, sampled_beatmaps_data, norms)

    # 5. Reduce dimensionality from D_MODEL to 2 for visualization
    print(f"Reducing {embeddings.shape[1]} dimensions to 2 using PCA...")
    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    print("Dimensionality reduction complete.")

    # 6. Create the interactive HTML graph
    graph_file = create_interactive_graph(reduced_embeddings, pks, pk_to_beatmap_id_map, pk_to_sr_map, OUTPUT_HTML_PATH)

    # 7. Automatically open the graph in the default web browser
    try:
        webbrowser.open('file://' + os.path.realpath(graph_file))
        print(f"Opening {graph_file} in your browser...")
    except Exception as e:
        print(f"Could not automatically open the file. Please open it manually: {os.path.realpath(graph_file)}")
        print(f"Error: {e}")

    print("\nProgram finished. Goodbye!")