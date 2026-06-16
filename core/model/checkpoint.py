from __future__ import annotations

import re
from typing import Any

import torch
from omegaconf import OmegaConf

def normalize_checkpoint_state(
    state: dict[str, torch.Tensor],
    flatten_difficulty_head: bool = True,
) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state.items():
        for prefix in ("model._orig_mod.", "model.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if flatten_difficulty_head and key.startswith("difficulty_head.head."):
            key = key.replace("difficulty_head.head.", "difficulty_head.", 1)
        normalized[key] = value
    return normalized


def configure_from_checkpoint(
    config: Any,
    checkpoint: dict[str, Any],
    state: dict[str, torch.Tensor],
    *,
    alignment: bool,
) -> Any:
    checkpoint_config = _checkpoint_config(checkpoint)
    OmegaConf.set_struct(config, False)
    if checkpoint_config is not None:
        config = OmegaConf.merge(config, checkpoint_config)
        OmegaConf.set_struct(config, False)

    _apply_encoder_shape(config, state)
    if alignment:
        _apply_alignment_shape(config, state)
    else:
        _apply_pretraining_shape(config, state)
    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, True)
    return config


def _checkpoint_config(checkpoint: dict[str, Any]) -> Any | None:
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict) or "config" not in hyper_parameters:
        return None
    return OmegaConf.create(
        OmegaConf.to_container(hyper_parameters["config"], resolve=True)
    )


def _apply_encoder_shape(config: Any, state: dict[str, torch.Tensor]) -> None:
    wqkv = state.get("bert.layers.0.self_attn.wqkv.weight")
    if wqkv is not None and len(wqkv.shape) == 2:
        config.model.d_model = int(wqkv.shape[1])

    layer_indices = [
        int(match.group(1))
        for key in state
        if (match := re.match(r"bert\.layers\.(\d+)\.self_attn\.wqkv\.weight$", key))
    ]
    if layer_indices:
        config.model.n_layers = max(layer_indices) + 1

    tokenizer_out = state.get("bert.feature_tokenizer.out.weight")
    if tokenizer_out is not None and len(tokenizer_out.shape) == 2:
        config.model.feature_token_dim = int(tokenizer_out.shape[1])

    w13 = state.get("bert.layers.0.ffn.w13.weight")
    if w13 is not None and len(w13.shape) == 2:
        config.model.dim_feedforward = int(w13.shape[0] // 2)


def _apply_pretraining_shape(config: Any, state: dict[str, torch.Tensor]) -> None:
    projection = state.get("difficulty_head.pooler.projections.mean.1.weight")
    if projection is not None and len(projection.shape) == 2:
        config.pretraining.pooling.stat_dim = int(projection.shape[0])


def _apply_alignment_shape(config: Any, state: dict[str, torch.Tensor]) -> None:
    query = state.get("contrastive_pooler.query")
    if query is not None and len(query.shape) == 3:
        config.alignment.query_pool.num_queries = int(query.shape[0])
        config.alignment.query_pool.heads = int(query.shape[1])
        config.alignment.query_pool.head_dim = int(query.shape[2])

    out = state.get("contrastive_pooler.out.1.weight")
    if out is not None and len(out.shape) == 2:
        config.alignment.query_pool.output_dim = int(out.shape[0])

    retrieval = state.get("retrieval_head.3.weight")
    if retrieval is not None and len(retrieval.shape) == 2:
        config.alignment.embedding_dim = int(retrieval.shape[0])

    projection = state.get("pooler.pooler.projections.mean.1.weight")
    if projection is None:
        projection = state.get("pooler.projections.mean.1.weight")
    if projection is not None and len(projection.shape) == 2:
        config.alignment.pooling.stat_dim = int(projection.shape[0])

    mixer = state.get("pooler.mixer.1.weight")
    if mixer is not None and len(mixer.shape) == 2:
        config.alignment.pooling.stats_mixer_dim = int(mixer.shape[0])

    map_projection = state.get("map_projector.net.1.weight")
    if map_projection is not None and len(map_projection.shape) == 2:
        config.alignment.map_features.dim = int(map_projection.shape[0])
