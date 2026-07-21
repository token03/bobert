from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

from ..data.normalizer import BeatmapNormalizer


def model_spec_from_config(config: DictConfig) -> dict[str, Any]:
    return {
        "data": {"max_seq_len": config.data.max_seq_len},
        "model": OmegaConf.to_container(config.model, resolve=True),
        "training": {
            "masking": OmegaConf.to_container(config.training.masking, resolve=True)
        },
    }


def setup_checkpoint(config: DictConfig, checkpoint: dict[str, Any]):
    if "state_dict" not in checkpoint:
        raise RuntimeError("BoBERT checkpoint must contain a state_dict.")
    model_spec = checkpoint.get("model_spec")
    if model_spec is None:
        raise RuntimeError("BoBERT checkpoint must contain model_spec metadata.")
    if OmegaConf.is_config(model_spec):
        model_spec = OmegaConf.to_container(model_spec, resolve=True)

    training_spec = model_spec.get("training") or model_spec.get("pretraining")
    if training_spec is None:
        raise RuntimeError("BoBERT checkpoint does not contain training metadata.")

    OmegaConf.set_struct(config, False)
    config.data.max_seq_len = model_spec["data"]["max_seq_len"]
    config.model = OmegaConf.merge(config.model, OmegaConf.create(model_spec["model"]))
    config.training.masking = OmegaConf.merge(
        config.training.masking, OmegaConf.create(training_spec["masking"])
    )
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
        stripped[key] = (
            value.detach().cpu() if isinstance(value, torch.Tensor) else value
        )
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

    compatible_state = {
        key: value for key, value in state.items() if key not in skipped
    }
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
    if normalizer is not None:
        checkpoint["vector_stats"] = normalizer.get_vector_stats()


def restore_normalizer_from_checkpoint(
    checkpoint: dict[str, Any], normalizer: BeatmapNormalizer | None
) -> None:
    if normalizer is not None and "vector_stats" in checkpoint:
        normalizer.vector_stats = checkpoint["vector_stats"]


def normalizer_from_checkpoint(checkpoint: dict[str, Any]) -> BeatmapNormalizer:
    return BeatmapNormalizer(vector_stats=checkpoint["vector_stats"])


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
