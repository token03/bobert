---
license: mit
tags:
- osu
- representation-learning
- similarity-search
---

# BoBERT · $revision

Hit-object sequence encoders and retrieval indices for osu! modes. Each mode has its own tokenizer, encoder, layer-centering statistics and collection-trained linear adapter, all in `model.safetensors`.

| Mode | Maps | Dimensions | Layers |
|---|---:|---:|---:|
$rows

`<mode>/embeddings/` holds append-only bfloat16 shards; a later shard replaces earlier rows of the same beatmap. Each shard records the checksum of the model that embedded it. `densities.parquet` holds the retrieval density of every indexed map, and `search.bin` the client search catalog. Metadata catalogs are in `data/`.

Export workspace: `$commit`.

From a source checkout:

```python
import torch
from core.model import BobertEncoder

model = BobertEncoder.from_pretrained(
    "runs/current/mania/model.safetensors", torch.device("cpu")
)
```

[Source, training and evaluation](https://github.com/token03/bobert)
