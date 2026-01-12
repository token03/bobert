# BoBERT + CM2T2A
**Bidirectional osu! Beatmap Encoder Representations from Transformers + Contrastive Map-Metadata-Tag-Topic Alignment**

## Model Architecture

### BoBERT (Beatmap Encoder)
Transformer encoder for learning beatmap representations from hit object sequences. Based on ModernBERT design principles.

**Architecture:**
* **Feature Embedding:** Hit object features grouped (spatial, rhythm, slider, categorical) and embedded via FT-Transformer-style layers.
* **Attention:** Alternating local/global attention (every 3rd layer global, others use sliding window).
* **Optimization:** RoPE positional embeddings, Flash Attention 2, variable-length packing.
* **Core Layers:** SwiGLU FFN, RMSNorm, no biases.

**Pretraining (Phase 1):**
1. **SMLM:** Span-masked language modeling on hit object sequences.
2. **Regression:** Difficulty attribute regression (`stars`, `aim`, `speed`, `slider_factor`, `AR`, `CS`, `slider_multiplier`).

---

### CM2T2A (Metadata-Tag-Topic Alignment)
Lightweight tabular transformer encoding beatmap metadata, user tags, and collection topics.

**Inputs:**
* **Metadata:** Piecewise linear encoding for continuous features (ratings, `AR`, `CS`, etc.); learned embeddings for categorical (mapper, genre, language, status, etc.).
* **User Tags:** Raw tags with vote counts from website (noisy, sparse, positive-unlabelled).
* **Collection Topics:** NMF-derived topics from collection co-occurrence patterns (see `scripts/nmf.py`).

**Alignment (Phase 2):**
* **Mechanism:** Contrastive learning between BoBERT embeddings $\leftrightarrow$ Tabformer embeddings.
* **Objective:** Align technical beatmap patterns with human sentiment and categorization.
* **Tasks:** Primary: beatmap-to-beatmap retrieval | Secondary: metadata/tag-based search.

## Use Cases
Semantic search, similarity/recommendation, quality/difficulty prediction — any understanding task (no generation).