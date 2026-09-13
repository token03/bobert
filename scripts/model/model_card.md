---
license: mit
tags:
- osu
- representation-learning
- similarity-search
---

# BoBERT · $revision

Dense osu!standard beatmap representations from `.osu` hit-object sequences.
The encoder was pretrained on roughly 500,000 beatmaps; collection-derived adaptation
refines the embeddings for recommendation and similarity search.

[Source and architecture](https://github.com/token03/bobert) · [Live recommender](https://bobert-web.pages.dev/)

## Release

- Embeddings: $count beatmaps, $dimensions dimensions.
- Encoder: $layers layers, $heads heads, $max_seq_len hit-object limit.
- Index generated: $generated_at.
- Adapter: $adapter.
- Export workspace: `$commit`.

## Download

From the source checkout:

```sh
uv run --no-default-groups --group serve fetch-run --repo $repo --revision $revision
docker compose up --build api
```

To inspect the model directly:

```python
import torch
from core.model import BobertEncoder

model, stats = BobertEncoder.from_pretrained("model.safetensors", torch.device("cpu"))
```

`model.safetensors` contains the encoder, adapter, normalization statistics, and model configuration.
`embeddings.parquet` contains beatmap IDs, vectors, densities, and embedded pooling/retrieval metadata.
The index records its model's SHA-256 checksum. Keep the two files together for online queries.
`data/` contains the metadata and strain catalogs required by the API.

See the source repository for the training pipeline, acknowledgements, limitations and evaluation tools.
