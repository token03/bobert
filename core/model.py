from pathlib import Path
from omegaconf import DictConfig
import torch
import torch.nn as nn
from typing import Any, Dict, Sequence, Type, TypeVar
from rotary_embedding_torch import RotaryEmbedding

from .components import (
    EncoderLayer,
    HitObjectFeatureTokenizer,
    MaskedLMHead,
    RMSNorm,
    SpanMasker,
)
from .features import FEATURE_INFO, VectorStats


T = TypeVar("T", bound="BobertEncoder")


class BobertEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        local_attention_window: int,
        local_attention_block_size: int,
        global_attention_layers: Sequence[int],
        dropout: float,
        max_seq_len: int,
        activation_checkpointing: bool,
        feature_token_dim: int,
        use_flash: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.activation_checkpointing = activation_checkpointing
        self.global_attention_layers = set(global_attention_layers)
        self.use_flash = use_flash
        self.max_seq_len = max_seq_len
        self.model_args = {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dim_feedforward": dim_feedforward,
            "local_attention_window": local_attention_window,
            "local_attention_block_size": local_attention_block_size,
            "global_attention_layers": list(global_attention_layers),
            "dropout": dropout,
            "max_seq_len": max_seq_len,
            "feature_token_dim": feature_token_dim,
        }

        self.feature_info = FEATURE_INFO
        self.feature_tokenizer = HitObjectFeatureTokenizer(
            feature_info=self.feature_info,
            d_feat=feature_token_dim,
            d_model=d_model,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    n_heads,
                    dim_feedforward,
                    dropout,
                    is_global=i in self.global_attention_layers,
                    local_window_size=local_attention_window,
                    local_block_size=local_attention_block_size,
                    activation_checkpointing=activation_checkpointing,
                    use_flash=use_flash,
                )
                for i in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

        self.rotary_emb = RotaryEmbedding(
            dim=d_model // n_heads, cache_max_seq_len=max_seq_len
        )

    @classmethod
    def from_config(cls: Type[T], config: DictConfig, *, use_flash: bool) -> T:
        model_config = config.model
        data_config = config.data
        runtime_config = config.runtime

        return cls(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
            local_attention_window=model_config.local_attention_window,
            local_attention_block_size=getattr(
                model_config, "local_attention_block_size", 256
            ),
            global_attention_layers=model_config.global_attention_layers,
            max_seq_len=data_config.max_seq_len,
            activation_checkpointing=runtime_config.activation_checkpointing,
            feature_token_dim=model_config.feature_token_dim,
            use_flash=use_flash,
        )

    @classmethod
    def from_pretrained(
        cls: Type[T], path: str | Path, device: torch.device
    ) -> tuple[T, VectorStats]:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            **artifact["model_args"],
            activation_checkpointing=False,
            use_flash=device.type == "cuda",
        )
        model.load_state_dict(artifact["state_dict"], strict=True)
        model.to(device).eval()
        return model, artifact["vector_stats"]

    def save_pretrained(
        self,
        path: str | Path,
        vector_stats: VectorStats,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        torch.save(
            {
                "model_args": self.model_args,
                "state_dict": state,
                "vector_stats": vector_stats,
            },
            path,
        )

    def get_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "parameter_efficiency": trainable_params / total_params,
        }

    def embed_sequences(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_tokenizer(x)

    def encode(
        self,
        packed_embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        positions = torch.arange(max_seqlen, device=packed_embeddings.device)
        torch._dynamo.mark_dynamic(packed_embeddings, 0)
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        torch._dynamo.mark_dynamic(positions, 0)
        return self._encode(packed_embeddings, cu_seqlens, positions)

    def _encode(
        self,
        packed_embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        packed_output = packed_embeddings
        max_seqlen = positions.shape[0]
        all_freqs = self.rotary_emb(
            positions,
            seq_len=max_seqlen,
        )
        if self.use_flash:
            rotary_freqs = (all_freqs[:, ::2].cos(), all_freqs[:, ::2].sin())
        else:
            total_tokens = packed_embeddings.shape[0]
            token_idx = torch.arange(total_tokens, device=packed_embeddings.device)
            batch_ids = torch.bucketize(token_idx, cu_seqlens[1:], right=True)
            pos = token_idx - cu_seqlens[batch_ids]
            rotary_freqs = all_freqs[pos].view(
                total_tokens, 1, self.d_model // self.n_heads
            )

        for layer in self.layers:
            packed_output = layer(
                packed_output,
                rotary_freqs=rotary_freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        packed_output = self.final_norm(packed_output)

        return packed_output

    def encode_packed(
        self,
        packed_vectors: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        torch._dynamo.mark_dynamic(packed_vectors, 0)
        packed_input = self.embed_sequences(packed_vectors)
        packed_output = self.encode(
            packed_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return packed_output, cu_seqlens, max_seqlen

    def _get_embedding(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        pooled = torch.segment_reduce(
            packed_output.float(), reduce="mean", lengths=lengths
        )
        return pooled.masked_fill(lengths[:, None] == 0, 0.0)

    def embed_packed(
        self,
        packed_vectors: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        packed_output, cu_seqlens, _ = self.encode_packed(
            packed_vectors, cu_seqlens, max_seqlen
        )
        return self._get_embedding(packed_output, cu_seqlens)


class BobertForPretraining(nn.Module):
    def __init__(
        self,
        bert_model: BobertEncoder,
        masker: SpanMasker,
        mlm_head: MaskedLMHead,
    ):
        super().__init__()
        self.bert = bert_model
        self.masker = masker
        self.mlm_head = mlm_head

    @classmethod
    def from_config(
        cls, config: DictConfig, device: torch.device
    ) -> "BobertForPretraining":
        bert = BobertEncoder.from_config(config, use_flash=device.type == "cuda")
        return cls(
            bert,
            SpanMasker(bert.d_model),
            MaskedLMHead(bert.d_model),
        ).to(device)

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def forward_packed(
        self,
        packed_vectors: torch.Tensor,
        masked_idx: torch.Tensor,
        mask_token_idx: torch.Tensor,
        random_dst_idx: torch.Tensor,
        right_border_zero_idx: torch.Tensor,
        right_border_random_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        packed_targets = {"mlm": packed_vectors.index_select(0, masked_idx)}
        encoder_vectors = self.masker.corrupt_inputs_packed(
            packed_vectors,
            right_border_zero_idx,
            right_border_random_idx,
        )
        torch._dynamo.mark_dynamic(encoder_vectors, 0)
        packed_embed = self.bert.embed_sequences(encoder_vectors)
        packed_input = self.masker.forward_packed(
            packed_embed,
            mask_token_idx,
            random_dst_idx,
        )
        encoded = self.bert.encode(
            packed_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        predictions = {"mlm": self.mlm_head(encoded.index_select(0, masked_idx))}
        return predictions, packed_targets, masked_idx
