# finetuner.py
import time
import math
from typing import Dict, Any, Optional, Tuple, List
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager
from .metrics import MetricsTracker, FineTuneEpochMetrics
from .optimization import create_optimizer, create_scheduler
from .loss import contrastive_loss_fn
from ..data.transforms import BeatmapNormalizer

class FineTuningTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        normalizer: BeatmapNormalizer,
        user_tag_encoder: Dict[str, int],
        collection_label_encoder: Dict[str, int]
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.normalizer = normalizer
        self.loss_fn = contrastive_loss_fn
        self.user_tag_encoder = user_tag_encoder
        self.collection_label_encoder = collection_label_encoder
        
        self.use_amp = config['pretraining'].get('use_amp', False) and device.type == 'cuda'
        self.grad_clip_norm = config['pretraining'].get('grad_clip_norm', 1.0)
        self.grad_accum_steps = config['pretraining'].get('gradient_accumulation_steps', 1)
        
        self.scaler = torch.amp.GradScaler(device=device.type, enabled=self.use_amp)
        self.metrics_tracker = MetricsTracker()
        self.val_metrics_computer = FineTuneEpochMetrics(config).to(device)

        print(f"FineTuningTrainer initialized - AMP: {self.use_amp}, Device: {device}, Grad Accum: {self.grad_accum_steps}")
        if self.train_dataloader.batch_sampler is not None:
            effective_batch_size = self.train_dataloader.batch_sampler.batch_size * self.grad_accum_steps
        else:
            effective_batch_size = self.train_dataloader.batch_size * self.grad_accum_steps
        print(f"Effective batch size: {effective_batch_size}")

    def _encode_labels(self, labels: List[List[str]], encoder: Dict[str, int], num_classes: int) -> torch.Tensor:
        batch_size = len(labels)
        encoded_tensor = torch.zeros(batch_size, num_classes, device=self.device)
        for i, sample_labels in enumerate(labels):
            if not sample_labels:
                continue
            for label in sample_labels:
                if label in encoder:
                    encoded_tensor[i, encoder[label]] = 1.0
        return encoded_tensor
    
    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        vectors, attention_mask, difficulty_ratings, collection_labels, user_tags, positive_mask = batch

        vectors = vectors.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        
        norm_difficulty_ratings = self.normalizer.normalize_difficulty(
            difficulty_ratings.to(self.device, non_blocking=True)
        )

        labels_dict = {
            'difficulty_ratings': norm_difficulty_ratings,
            'positive_mask': positive_mask.to(self.device, non_blocking=True)
        }
        
        if any(collection_labels):
            num_collection_classes = self.model.module.collection_label_head.out_features if isinstance(self.model, nn.DataParallel) else self.model.collection_label_head.out_features
            encoded_collections = self._encode_labels(collection_labels, self.collection_label_encoder, num_collection_classes)
            labels_dict['collection_labels'] = encoded_collections

        is_user_tag_head_present = hasattr(self.model, 'user_tag_head') or (isinstance(self.model, nn.DataParallel) and hasattr(self.model.module, 'user_tag_head'))
        if is_user_tag_head_present and any(user_tags):
            num_user_tag_classes = self.model.module.user_tag_head.out_features if isinstance(self.model, nn.DataParallel) else self.model.user_tag_head.out_features
            encoded_tags = self._encode_labels(user_tags, self.user_tag_encoder, num_user_tag_classes)
            labels_dict['user_tags'] = encoded_tags
            
        return vectors, attention_mask, labels_dict

    def _run_step(self, batch: Tuple, is_train: bool) -> Dict[str, float]:
        vectors, attention_mask, labels = self._prepare_batch(batch)
        
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                predictions = self.model(vectors, attention_mask)
                losses = self.loss_fn(predictions, labels, self.config)
        
        if is_train:
            scaled_loss = losses['total_loss'] / self.grad_accum_steps
            self.scaler.scale(scaled_loss).backward()
        
        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        epoch_losses = {}
        
        num_update_steps = math.ceil(len(self.train_dataloader) / self.grad_accum_steps)
        progress_bar = tqdm(
            total=num_update_steps,
            desc=f"Epoch {epoch+1} [Train]",
            dynamic_ncols=True
        )
        
        data_iter = iter(self.train_dataloader)
        for i in range(len(self.train_dataloader)):
            batch = next(data_iter)
            step_losses = self._run_step(batch, is_train=True)
            
            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            
            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.train_dataloader):
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.scheduler: self.scheduler.step()
                
                progress_bar.update(1)

            progress_bar.set_postfix({
                "Loss": f"{step_losses['total_loss']:.4f}",
                "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
        progress_bar.close()
        avg_losses = {k: v / len(self.train_dataloader) for k, v in epoch_losses.items()}
        avg_losses['learning_rate'] = self.optimizer.param_groups[0]['lr']
        return avg_losses

    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        self.val_metrics_computer.reset_batch_metrics()

        all_embeddings = []
        all_ratings = [] 
        all_labels = []

        print("Running validation...")
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False, dynamic_ncols=True):
                vectors, attention_mask, ratings, labels, _, _ = batch

                vectors_dev = vectors.to(self.device, non_blocking=True)
                attention_mask_dev = attention_mask.to(self.device, non_blocking=True)

                norm_ratings_dev = self.normalizer.normalize_difficulty(
                    ratings.to(self.device, non_blocking=True)
                )
                
                labels_dict = {
                    'difficulty_ratings': norm_ratings_dev
                }

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    predictions = self.model(vectors_dev, attention_mask_dev)
                    step_losses = self.loss_fn(predictions, labels_dict, self.config)
                    embeddings = predictions.get('collection_label_projection', predictions['sequence_representation'])

                self.val_metrics_computer.update_batch_metrics({k: v.item() for k,v in step_losses.items()})

                all_embeddings.append(embeddings.cpu())
                all_ratings.append(ratings.cpu()) 
                all_labels.extend(labels)

        batch_metrics = self.val_metrics_computer.compute_batch_metrics()

        embedding_metrics = self.val_metrics_computer.compute_embedding_metrics(
            all_embeddings=torch.cat(all_embeddings, dim=0).to(self.device),
            all_ratings=torch.cat(all_ratings, dim=0).to(self.device),
            all_labels=all_labels
        )
        
        all_metrics = {**batch_metrics, **embedding_metrics}
        return all_metrics

    def train(self, start_epoch: int = 0):
        num_epochs = self.config['finetuning']['num_epochs']
        print(f"Starting fine-tuning from epoch {start_epoch+1}/{num_epochs}...")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch()
            
            self.metrics_tracker.log_epoch(epoch, train_metrics, val_metrics)
            
            epoch_duration = time.time() - epoch_start_time
            
            stats_to_save = {
                'vector_stats': self.normalizer.get_vector_stats(),
                'difficulty_stats': self.normalizer.get_difficulty_stats(),
            }
            
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                epoch, val_metrics, suffix="latest",
                stats=stats_to_save 
            )
            r1 = val_metrics.get('Recall@1', 0.0)
            r5 = val_metrics.get('Recall@5', 0.0)
            r10 = val_metrics.get('Recall@10', 0.0)
            rho = val_metrics.get('SpearmanRho', 0.0)
            ndcg10 = val_metrics.get('nDCG@10', 0.0)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_duration:.2f}s | "
                  f"Train Loss: {train_metrics['total_loss']:.4f} | "
                  f"Val Loss: {val_metrics.get('total_loss', 0.0):.4f} | "
                  f"R@1: {r1:.3f} | R@5: {r5:.3f} | R@10: {r10:.3f} | Rho: {rho:.3f} | nDCG@10: {ndcg10:.3f}")
        
        print("Fine-tuning finished.")
        return self.metrics_tracker


def setup_finetuning(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    normalizer: BeatmapNormalizer,
    user_tag_encoder: Dict[str, int],
    collection_label_encoder: Dict[str, int]
) -> Tuple[FineTuningTrainer, CheckpointManager]:
    def _resolve_ratings(dataset) -> Optional[List[float]]:
        if hasattr(dataset, 'ratings'):
            return dataset.ratings
        nested_dataset = getattr(dataset, 'dataset', None)
        if nested_dataset is None:
            return None
        base_ratings = _resolve_ratings(nested_dataset)
        if base_ratings is None:
            return None
        indices = getattr(dataset, 'indices', None)
        if indices is None:
            return base_ratings
        return [base_ratings[i] for i in indices]

    grad_accum_steps = config['finetuning'].get('gradient_accumulation_steps', 1)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    total_steps = num_update_steps_per_epoch * config['finetuning']['num_epochs']

    optimizer = create_optimizer(model, config, 'finetuning')
    scheduler = create_scheduler(optimizer, config, total_steps, 'finetuning')

    checkpoint_dir = config['finetuning']['checkpoint_dir']
    model_name = config['model'].get('type', 'model') + "_finetuned"
    checkpoint_manager = CheckpointManager(checkpoint_dir, model_name)
    
    if normalizer.get_difficulty_stats() is None:
        ratings = _resolve_ratings(train_dataloader.dataset)
        if ratings is None:
            raise ValueError(
                "BeatmapNormalizer must be provided with difficulty statistics for fine-tuning, "
                "but they were missing and the training dataset does not expose raw difficulty ratings."
            )
        normalizer.update_difficulty_stats(torch.as_tensor(ratings, dtype=torch.float32))

    trainer = FineTuningTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        checkpoint_manager=checkpoint_manager,
        normalizer=normalizer,
        user_tag_encoder=user_tag_encoder,
        collection_label_encoder=collection_label_encoder
    )
    
    return trainer, checkpoint_manager