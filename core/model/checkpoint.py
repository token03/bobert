from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

from ..data.normalizer import BeatmapNormalizer


def model_spec_from_config(config: DictConfig, phase: str) -> dict[str, Any]:
    spec = {
        "data": {
            "max_seq_len": config.data.max_seq_len,
        },
        "model": OmegaConf.to_container(config.model, resolve=True),
    }
    if phase == "alignment":
        spec["alignment"] = {
            "embedding_dim": config.alignment.embedding_dim,
            "pooling": OmegaConf.to_container(config.alignment.pooling, resolve=True),
            "query_pool": OmegaConf.to_container(config.alignment.query_pool, resolve=True),
            "map_features": OmegaConf.to_container(config.alignment.map_features, resolve=True),
        }
    elif phase == "pretraining":
        spec["pretraining"] = {
            "pooling": OmegaConf.to_container(config.pretraining.pooling, resolve=True),
            "masking": OmegaConf.to_container(config.pretraining.masking, resolve=True),
        }
    elif phase == "adapter":
        spec["adapter"] = OmegaConf.to_container(config.adapter, resolve=True)
    else:
        raise ValueError(f"Unsupported checkpoint phase: {phase}")
    return spec


def setup_checkpoint(config: DictConfig, checkpoint: dict[str, Any], phase: str):
    if "state_dict" not in checkpoint:
        raise RuntimeError("BoBERT checkpoint must contain a state_dict.")

    model_spec = checkpoint.get("model_spec")
    if model_spec is None:
        raise RuntimeError("BoBERT checkpoint must contain model_spec metadata.")
    if OmegaConf.is_config(model_spec):
        model_spec = OmegaConf.to_container(model_spec, resolve=True)

    OmegaConf.set_struct(config, False)
    config.data.max_seq_len = model_spec["data"]["max_seq_len"]
    config.model = OmegaConf.merge(config.model, OmegaConf.create(model_spec["model"]))
    if phase == "alignment":
        phase_spec = model_spec["alignment"]
        config.alignment.embedding_dim = phase_spec["embedding_dim"]
        config.alignment.pooling = OmegaConf.merge(
            config.alignment.pooling, OmegaConf.create(phase_spec["pooling"])
        )
        config.alignment.query_pool = OmegaConf.merge(
            config.alignment.query_pool, OmegaConf.create(phase_spec["query_pool"])
        )
        config.alignment.map_features = OmegaConf.merge(
            config.alignment.map_features, OmegaConf.create(phase_spec["map_features"])
        )
    elif phase == "pretraining":
        phase_spec = model_spec["pretraining"]
        config.pretraining.pooling = OmegaConf.merge(
            config.pretraining.pooling, OmegaConf.create(phase_spec["pooling"])
        )
        config.pretraining.masking = OmegaConf.merge(
            config.pretraining.masking, OmegaConf.create(phase_spec["masking"])
        )
    elif phase == "adapter":
        config.adapter = OmegaConf.merge(
            config.adapter, OmegaConf.create(model_spec["adapter"])
        )
    else:
        raise ValueError(f"Unsupported checkpoint phase: {phase}")
    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, True)
    return config, checkpoint["state_dict"]


def strip_checkpoint_state(state: dict[str, Any]) -> dict[str, Any]:
    stripped = {}
    for key, value in state.items():
        for prefix in ("model._orig_mod.", "model.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        stripped[key] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    return stripped


def load_state_for_inference(
    model: nn.Module, state: dict[str, Any]
) -> torch.nn.modules.module._IncompatibleKeys:
    target = getattr(model, "_orig_mod", model)
    model_state = target.state_dict()
    optional_keys = {"masker.span_length_probs", "masker.span_lengths_range"}
    skipped = {
        key
        for key in optional_keys
        if key in state
        and key in model_state
        and tuple(model_state[key].shape) != tuple(state[key].shape)
    }
    if not skipped:
        return target.load_state_dict(state, strict=True)

    compatible_state = {key: value for key, value in state.items() if key not in skipped}
    missing, unexpected = target.load_state_dict(compatible_state, strict=False)
    if set(missing) != skipped or unexpected:
        raise RuntimeError(
            "Pretraining checkpoint is not compatible with this model; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)


def normalize_lightning_state_dict(
    state_dict: dict[str, Any], model_is_compiled: bool
) -> dict[str, Any]:
    normalized_state = {}
    for key, value in state_dict.items():
        if model_is_compiled:
            if key.startswith("model.") and not key.startswith("model._orig_mod."):
                key = "model._orig_mod." + key[len("model.") :]
        elif key.startswith("model._orig_mod."):
            key = "model." + key[len("model._orig_mod.") :]
        normalized_state[key] = value
    return normalized_state


def add_normalizer_to_checkpoint(
    checkpoint: dict[str, Any], normalizer: BeatmapNormalizer | None
) -> None:
    if normalizer is None:
        return
    checkpoint["vector_stats"] = normalizer.get_vector_stats()
    checkpoint["attribute_stats"] = normalizer.get_attribute_stats()


def restore_normalizer_from_checkpoint(
    checkpoint: dict[str, Any], normalizer: BeatmapNormalizer | None
) -> None:
    if normalizer is None:
        return
    if "vector_stats" in checkpoint:
        normalizer.vector_stats = checkpoint["vector_stats"]
    if "attribute_stats" in checkpoint:
        normalizer.attribute_stats = checkpoint["attribute_stats"]


def normalizer_from_checkpoint(checkpoint: dict[str, Any]) -> BeatmapNormalizer:
    return BeatmapNormalizer(
        vector_stats=checkpoint["vector_stats"],
        attribute_stats=checkpoint.get("attribute_stats", {}),
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def load_pretraining_checkpoint(
    checkpoint_path: str | Path | None,
    map_location: str | torch.device = "cpu",
) -> tuple[Path, dict[str, Any]]:
    if checkpoint_path is None:
        raise FileNotFoundError("No pretraining checkpoint found.")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    if "vector_stats" not in checkpoint:
        raise RuntimeError(
            "Pretraining checkpoint does not contain vector_stats; "
            "alignment requires the pretraining normalizer."
        )
    return checkpoint_path, checkpoint


def load_pretraining_weights(
    model: nn.Module,
    checkpoint_path: str | Path | None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path, checkpoint = load_pretraining_checkpoint(checkpoint_path, map_location)
    state = strip_checkpoint_state(checkpoint["state_dict"])
    target = getattr(model, "_orig_mod", model)
    model_state = target.state_dict()
    compatible_state = {
        key: value
        for key, value in state.items()
        if key.startswith("bert.")
        and key in model_state
        and tuple(model_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = target.load_state_dict(compatible_state, strict=False)
    skipped = sorted(set(state) - set(compatible_state))
    tokenizer_keys = {key for key in model_state if key.startswith("bert.feature_tokenizer.")}
    missing_tokenizer_keys = sorted(tokenizer_keys - set(compatible_state))
    if missing_tokenizer_keys:
        raise RuntimeError(
            "Pretraining checkpoint does not contain compatible feature tokenizer weights; "
            "alignment requires a pretrained tokenizer."
        )

    return {
        "checkpoint_path": checkpoint_path,
        "loaded": len(compatible_state),
        "skipped": len(skipped),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "loaded_tokenizer": len(tokenizer_keys),
    }


def load_pretraining_normalizer(
    checkpoint_path: str | Path | None,
    map_location: str | torch.device = "cpu",
) -> BeatmapNormalizer:
    _, checkpoint = load_pretraining_checkpoint(checkpoint_path, map_location)
    return normalizer_from_checkpoint(checkpoint)
