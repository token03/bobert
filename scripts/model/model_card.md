---
license: mit
tags:
- osu
- representation-learning
- similarity-search
---

# BoBERT · $revision

Hit-object sequence encoders and retrieval indices for osu! modes. Each mode has its own tokenizer, encoder weights, centering statistics and optional collection-trained linear adapter.

| Mode | Variant | Maps | Dimensions | Layers |
|---|---|---:|---:|---:|
$rows

The release manifest selects one model/index pair per mode. Base and adapted files use the same mode directory; adapted files have an `.a` suffix. The index records its model's SHA256 checksum. Catalogs are shared in `data/`.

Export workspace: `$commit`.

```sh
uv run fetch-run --repo $repo --revision $revision
docker compose up --build api
```

From a source checkout:

```python
import torch
from core.model import BobertEncoder
from core.artifacts import resolve_artifacts

artifacts = resolve_artifacts("runs/current", "mania")
model = BobertEncoder.from_pretrained(artifacts.model, torch.device("cpu"))
```

[Source, training and evaluation](https://github.com/token03/bobert)
