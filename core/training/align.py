from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from core.training.metrics import ContrastiveMetrics

from .base import BobertLightningModule
from .loss import alignment_loss_fn
from ..data.normalizer import BeatmapNormalizer


class AlignmentModule(BobertLightningModule):
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        normalizer: Optional[BeatmapNormalizer] = None,
    ):
        super().__init__("alignment")
        self.model = model
        self.config = config
        self.normalizer = normalizer
        self.save_hyperparameters(ignore=["model", "normalizer"])

        self.batch_size = config.alignment.trainer.batch_size
        self.metrics = ContrastiveMetrics(torch.device("cpu"))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def on_fit_start(self):
        if self.global_rank == 0:
            print("Running warmup pass to initialize RoPE cache to max_seq_len...")

        max_seq_len = self.config["data"]["max_seq_len"]
        target_dtype = torch.float32

        precision_str = str(self.trainer.precision)
        if "bf16" in precision_str:
            target_dtype = torch.bfloat16
        elif "16" in precision_str:
            target_dtype = torch.float16

        model_to_run = self.model
        if hasattr(model_to_run, "_orig_mod"):
            model_to_run = model_to_run._orig_mod

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=target_dtype):
                model_to_run.bert.rotary_emb(
                    torch.arange(max_seq_len, device=self.device), seq_len=max_seq_len
                )

        if self.global_rank == 0:
            print(
                f"Warmup complete. RoPE cache initialized for L={max_seq_len} using {target_dtype}."
            )

    def _forward_packed_batch(self, batch: Dict[str, Any]):
        labels = batch["labels"]
        max_seqlen = int(batch["max_seqlen"].item())
        predictions = self.model.forward_packed(
            batch["packed_vectors"],
            batch["cu_seqlens"],
            max_seqlen,
            labels.get("map_features"),
        )
        return predictions, labels

    def _forward_batch(self, batch: Dict[str, Any]):
        if "packed_vectors" in batch:
            return self._forward_packed_batch(batch)
        labels = batch["labels"]
        predictions = self(
            batch["vectors"],
            batch["attention_mask"],
            batch["cu_seqlens"],
            labels.get("map_features"),
        )
        return predictions, labels

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        predictions, labels = self._forward_batch(batch)
        loss_dict = alignment_loss_fn(predictions, labels, self.config)

        self.log_dict(
            {
                "train_loss": loss_dict["total_loss"],
                "train_contrastive_loss": loss_dict["contrastive_loss"],
            },
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return loss_dict["total_loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        predictions, labels = self._forward_batch(batch)
        loss_dict = alignment_loss_fn(predictions, labels, self.config)
        self.metrics.update(loss=float(loss_dict["total_loss"].detach().cpu()))
        self.log(
            "val_loss",
            loss_dict["total_loss"],
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss_dict["total_loss"]

    def on_validation_epoch_end(self):
        results = self.metrics.compute()
        for key, value in results.items():
            self.log(f"val_{key}", value)
        self.metrics.reset()


def load_pretraining_checkpoint(
    checkpoint_path: str | Path | None,
    map_location: str | torch.device = "cpu",
) -> tuple[Path, Dict[str, Any]]:
    if checkpoint_path is None:
        raise FileNotFoundError("No pretraining checkpoint found.")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
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
) -> Dict[str, Any]:
    checkpoint_path, checkpoint = load_pretraining_checkpoint(checkpoint_path, map_location)
    raw_state = checkpoint.get("state_dict", checkpoint)
    state = {}

    for key, value in raw_state.items():
        for prefix in ("model._orig_mod.", "model.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.startswith("difficulty_head.pooler."):
            key = key.replace("difficulty_head.pooler.", "pooler.", 1)
        if key.startswith("difficulty_head.head."):
            key = key.replace("difficulty_head.head.", "difficulty_head.", 1)
        state[key] = value

    target = getattr(model, "_orig_mod", model)
    model_state = target.state_dict()
    compatible_state = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = target.load_state_dict(compatible_state, strict=False)
    skipped = sorted(set(state) - set(compatible_state))
    tokenizer_keys = {
        key for key in model_state if key.startswith("bert.feature_tokenizer.")
    }
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
    return BeatmapNormalizer(
        vector_stats=checkpoint["vector_stats"],
        attribute_stats=checkpoint.get("attribute_stats", {}),
    )
