# BoBERT
**Bidirectional osu! Beatmap Encoder Representations from Transformers**

## Model Architecture

### BoBERT (Beatmap Encoder)
Transformer encoder for learning beatmap representations from hit object sequences. Based on ModernBERT design principles.

**Architecture:**
* **Feature Embedding:** Hit object features grouped (spatial, rhythm, slider, categorical) and embedded via FT-Transformer-style layers.
* **Attention:** Alternating local/global attention (every 3rd layer global, others use sliding window).
* **Optimization:** RoPE positional embeddings, Flash Attention 2, variable-length packing.
* **Core Layers:** SwiGLU FFN, RMSNorm, no biases.

**Training:** Span-masked language modeling on hit object sequences.

## Use Cases
Semantic search and similarity/recommendation for beatmaps.
