# BoBERT
**Bidirectional osu! Beatmap Encoder Representations from Transformers**

BoBERT encodes osu! beatmaps as fixed-size vectors. It supports osu!standard, taiko, catch and mania, with one encoder for each ruleset. Each encoder reads the hit objects of a map and outputs one vector. An adapter, trained on collection data, then tunes each encoder for recommendation and similarity search.

The repository contains the data pipeline, the training and adaptation code, the evaluation tools and the web application.

**[Try the search tool](https://bobert-web.pages.dev/)**

## Architecture

BoBERT is a bidirectional Transformer encoder based on [ModernBERT](https://arxiv.org/abs/2412.13663). Each hit object is one token. A tokenizer in the style of [FT-Transformer](https://arxiv.org/abs/2106.11959) embeds the features of each hit object. Each ruleset has its own feature set. The encoder alternates local and global attention. The beatmap vector is the mean of the outputs of the global layers.

![BoBERT architecture: a per-ruleset feature tokenizer embeds each hit object's feature groups as 16-d vectors, concatenates them and projects them to a 384-d token. Tokens pass through nine pre-norm attention and SwiGLU blocks, with ±128-object local attention except at layers 3, 6 and 9, which attend globally. Those three layers are mean-pooled and passed through a linear adapter to give the 384-d beatmap vector.](docs/assets/architecture.png)

### Encoder

| Setting | Default in [`configs/pretrain.yaml`](configs/pretrain.yaml) |
| --- | --- |
| Parameters | 16.1M |
| Depth / width / heads | 9 layers / 384 / 6 |
| Sequence length limit | 4,096 hit objects |
| Attention | Global at layers 3, 6 and 9; ±128 hit objects at all other layers |

### Data

| Ruleset | Pretraining maps | Search index |
| --- | ---: | ---: |
| osu!standard | 692k | 499k |
| osu!taiko | 78k | 95k |
| osu!catch | 20k | 25k |
| osu!mania | 278k | 315k |

### Pretraining

- **Masked reconstruction.** The encoder reads the unmasked 70% of hit objects. A small training decoder reconstructs masked features. Features at span borders are also corrupted.
- **Strain regression.** A linear head predicts difficulty targets from the pooled vector. The targets are skill strains for osu!standard and star rating for the other rulesets.
- **Section contrast.** Pooled representations from alternating eight-event sections of a map are trained to agree.

### Adaptation

A linear adapter transforms the pretrained vector. The adapter starts as the identity. Contrastive training on pairs of maps that share collections then updates it. If a file of mined pairs is available, a second phase trains on these pairs.

The pooling statistics and the adapter are stored in the model, so a map's vector depends only on the model and the map. New maps can be embedded and appended to an index without changing existing vectors.

## Limitations

- The model reads hit objects only. It does not read AR, CS, OD or HP, so the vectors do not include these settings.
- Maps with few hit objects give the model less data, so their nearest neighbors are less reliable.
- The encoder reads the first 4,096 hit objects only.
- 1/3 and 1/4 rhythms are in different regions of the vector space. Thus, the same stream in 1/3 and in 1/4 is not similar.
- The rulesets share the architecture but not the weights, tokenizer, heads or normalization. Search compares maps of the same ruleset only.

## Run locally

Download a release from [Hugging Face](https://huggingface.co/token03/bobert):

```sh
uvx --from huggingface_hub hf download token03/bobert --revision v16 --local-dir runs/v16
mv runs/v16/data/* data/ && ln -sfn v16 runs/current
```

```text
data/                  beatmaps, beatmapsets and strains catalogs
runs/current -> v16/
  {std,taiko,catch,mania}/
    model.safetensors
    embeddings/*.parquet
    densities.parquet
    labels.safetensors
    search.bin
    training.yaml
```

Copy [`.env.example`](.env.example) to `.env` and fill in the osu! OAuth values and `SESSION_SECRET`. Install the frontend with `bun install` in `web/`, then start the API and Vite together:

```sh
mise run dev
```

## Training

The training environment targets Linux x86-64, Python 3.12, PyTorch 2.11 and CUDA 12.8:

```sh
uv sync --frozen
uv run pretrain --mode std --profile full -v my-run-base
uv run embed my-run-base --mode std
uv run adapt my-run-base my-run --mode std
```

The [scripts guide](scripts/README.md) covers every command, including updates and releases.

## Repository guide

| Directory | Contents |
| --- | --- |
| [`core/`](core/) | Beatmap parsing, features, encoder and dataset primitives |
| [`training/`](training/) | Lightning modules, data loading, objectives and optimizers |
| [`scripts/`](scripts/README.md) | Data download, feature extraction, training entry points and evaluation |
| [`server/`](server/README.md) | FastAPI application, retrieval runtime and embedding cache |
| [`web/`](web/README.md) | React application and Cloudflare gateway |
| `data/`, `runs/`, `cache/` | Local datasets, model artifacts and runtime state |

## Acknowledgements

- [MusicBERT](https://arxiv.org/abs/2106.05630): representation learning from symbolic music, a major inspiration.
- [CM3P](https://github.com/OliBomby/CM3P) by OliBomby: inspiration for the encoder architecture.
- [ModernBERT](https://github.com/AnswerDotAI/ModernBERT): bidirectional encoders with rotary embeddings and alternating attention.
- [FT-Transformer](https://github.com/yandex-research/rtdl-revisiting-models): feature-wise embeddings for numerical and categorical inputs.
- [parsecore](https://github.com/Apart-Studio/parsecore): star ratings and strain values for the pretraining targets.
- [osu!collector](https://osucollector.com/) and [osu!stats](https://osustats.ppy.sh/): collection data for adaptation and evaluation.
- [Beatconnect](https://beatconnect.io/): `.osu` files for the dataset.
- [osu!](https://osu.ppy.sh/) and its mapping community: the beatmaps and metadata.

## License

The code is available under the [MIT License](LICENSE).
