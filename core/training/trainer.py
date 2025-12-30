import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from typing import Dict, Any, List, Optional


def setup_device() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"

def create_trainer(
    config: Dict[str, Any],
    phase: str,
    checkpoint_dir: Optional[str] = None,
    extra_callbacks: Optional[List[pl.Callback]] = None,
) -> pl.Trainer:
    phase_config = config[phase]
    save_dir = checkpoint_dir or phase_config["checkpoint_dir"]

    progress_bar = TQDMProgressBar(refresh_rate=1)

    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,
            filename=f"{phase}-{{epoch:02d}}-{{val_loss:.4f}}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        progress_bar,
    ]

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    use_amp = phase_config.get("use_amp", False)
    precision = "bf16-mixed" if use_amp else 32

    return pl.Trainer(
        max_epochs=phase_config["num_epochs"],
        accelerator=setup_device(),
        devices=1,
        precision=precision,
        gradient_clip_val=phase_config.get("grad_clip_norm", 1.0),
        accumulate_grad_batches=phase_config.get("gradient_accumulation_steps", 1),
        logger=CSVLogger(save_dir=save_dir, name="logs"),
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
