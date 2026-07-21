# bobert.py
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Sequence, Type, TypeVar
from rotary_embedding_torch import RotaryEmbedding

from ..data.schema import FEATURE_INFO

from .components import (
    EncoderLayer,
    HitObjectFeatureTokenizer,
    MaskedLMHead,
    RMSNorm,
    SpanMasker,
)


T = TypeVar("T", bound="BobertEncoder")


def _compile_encoder_only(model: nn.Module, config: DictConfig, label: str) -> None:
    print(f"Compiling BERT {label} tokenizer and encoder with torch.compile...")
    compile_mode = config.runtime.compile_mode
    compile_dynamic = config.runtime.compile_dynamic
    model.bert.embed_sequences = torch.compile(
        model.bert.embed_sequences,
        mode=compile_mode,
        dynamic=compile_dynamic,
    )
    model.bert.encode = torch.compile(
        model.bert.encode,
        mode=compile_mode,
        dynamic=compile_dynamic,
    )
    model.bert.encode_masked = torch.compile(
        model.bert.encode_masked,
        mode=compile_mode,
        dynamic=compile_dynamic,
    )
    model.is_compiled = True


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
                    activation_checkpointing=activation_checkpointing and i % 2 == 0,
                    use_flash=use_flash,
                )
                for i in range(n_layers)
            ]
        )
        if self.layers:
            self.layers[-1].activation_checkpointing = False

        self.final_norm = RMSNorm(d_model)

        self.rotary_emb = RotaryEmbedding(
            dim=d_model // n_heads, cache_max_seq_len=max_seq_len
        )

    @classmethod
    def from_config(cls: Type[T], config: DictConfig, *, use_flash: bool) -> T:
        model_config = config.model
        data_config = config.data
        runtime_config = config.runtime

        dim_feedforward = model_config.dim_feedforward

        return cls(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dim_feedforward=dim_feedforward,
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
        packed_output = packed_embeddings
        all_freqs = self.rotary_emb(
            torch.arange(max_seqlen, device=packed_embeddings.device),
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

    def encode_masked(
        self,
        packed_embeddings: torch.Tensor,
        masked_idx: torch.Tensor,
        masked_positions: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        if not self.use_flash or not self.layers[-1].is_global:
            return self.encode(
                packed_embeddings,
                cu_seqlens_k,
                max_seqlen_k,
            ).index_select(0, masked_idx)

        all_freqs = self.rotary_emb(
            torch.arange(max_seqlen_k, device=packed_embeddings.device),
            seq_len=max_seqlen_k,
        )
        rotary_freqs = (all_freqs[:, ::2].cos(), all_freqs[:, ::2].sin())
        packed_output = packed_embeddings
        for layer in self.layers[:-1]:
            packed_output = layer(
                packed_output,
                rotary_freqs=rotary_freqs,
                cu_seqlens=cu_seqlens_k,
                max_seqlen=max_seqlen_k,
            )

        masked_output = self.layers[-1].forward_masked(
            packed_output,
            masked_idx=masked_idx,
            masked_positions=masked_positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            rotary_freqs=rotary_freqs,
        )
        return self.final_norm(masked_output)

    def encode_packed(
        self,
        packed_vectors: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        packed_input = self.embed_sequences(packed_vectors)
        packed_output = self.encode(
            packed_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return packed_output, cu_seqlens, max_seqlen


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
        self.is_compiled = False

    @classmethod
    def from_config(
        cls, config: DictConfig, device: torch.device
    ) -> "BobertForPretraining":
        use_flash = device.type == "cuda"
        base_model = BobertEncoder.from_config(config, use_flash=use_flash)
        training_config = config.training

        masking_strategy = SpanMasker(
            d_model=base_model.d_model,
            masking_ratio=training_config.masking.ratio,
            mean_span_length=training_config.masking.mean_span_length,
        )

        mlm_head = MaskedLMHead(base_model.d_model)
        model = cls(base_model, masking_strategy, mlm_head)
        model = model.to(device)

        if config.runtime.compile_model:
            _compile_encoder_only(model, config, "pre-training")

        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def _get_embedding(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
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
        packed_output, cu_seqlens, max_seqlen = self.bert.encode_packed(
            packed_vectors, cu_seqlens, max_seqlen
        )
        return self._get_embedding(packed_output, cu_seqlens, max_seqlen)

    def embed_col_packed(
        self,
        packed_vectors: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        beat_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        packed_output, cu_seqlens, max_seqlen = self.bert.encode_packed(
            packed_vectors, cu_seqlens, max_seqlen
        )
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        map_index = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=packed_output.device), lengths
        )
        token_index = torch.arange(
            packed_output.shape[0], device=packed_output.device
        ) - torch.repeat_interleave(cu_seqlens[:-1].to(torch.long), lengths)
        col_embedding = packed_output
        col_map_index = map_index
        col_index = token_index
        if beat_ids is not None:
            beat_ids = beat_ids.to(device=packed_output.device, dtype=torch.long)
            if beat_ids.shape[0] != packed_output.shape[0]:
                raise ValueError(
                    f"beat_ids length must match packed tokens: "
                    f"{beat_ids.shape[0]} != {packed_output.shape[0]}"
                )
            starts = torch.ones(
                beat_ids.shape[0], device=beat_ids.device, dtype=torch.bool
            )
            starts[1:] = (beat_ids[1:] != beat_ids[:-1]) | (
                map_index[1:] != map_index[:-1]
            )
            group_ids = starts.cumsum(0) - 1
            group_count = int(group_ids[-1].item()) + 1 if group_ids.numel() else 0
            col_embedding = torch.zeros(
                group_count,
                packed_output.shape[-1],
                device=packed_output.device,
                dtype=torch.float32,
            )
            col_embedding.index_add_(0, group_ids, packed_output.float())
            col_map_index = map_index[starts]
            col_index = beat_ids[starts]
        return {
            "embedding": self._get_embedding(packed_output, cu_seqlens, max_seqlen),
            "col_embedding": col_embedding,
            "col_map_index": col_map_index,
            "col_index": col_index,
        }

    def _predictions(
        self,
        packed_input: torch.Tensor,
        masked_idx: torch.Tensor,
        masked_positions: torch.Tensor,
        masked_counts: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens: torch.Tensor,
        max_seqlen_k: int,
    ) -> Dict[str, Any]:
        cu_seqlens_q = F.pad(
            torch.cumsum(masked_counts, dim=0, dtype=torch.int32), (1, 0)
        )
        masked_output = self.bert.encode_masked(
            packed_input,
            masked_idx,
            masked_positions,
            cu_seqlens_q,
            cu_seqlens,
            max_seqlen_q,
            max_seqlen_k,
        )
        return {"mlm": self.mlm_head(masked_output)}

    def forward_packed(
        self,
        packed_vectors: torch.Tensor,
        masked_idx: torch.Tensor,
        masked_positions: torch.Tensor,
        masked_counts: torch.Tensor,
        max_seqlen_q: int,
        mask_token_idx: torch.Tensor,
        random_dst_idx: torch.Tensor,
        right_border_zero_idx: torch.Tensor,
        right_border_random_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], torch.Tensor]:
        packed_targets = {"mlm": packed_vectors.index_select(0, masked_idx)}

        encoder_vectors = self.masker.corrupt_inputs_packed(
            packed_vectors,
            right_border_zero_idx,
            right_border_random_idx,
        )
        packed_embed = self.bert.embed_sequences(encoder_vectors)
        packed_input = self.masker.forward_packed(
            packed_embed,
            mask_token_idx,
            random_dst_idx,
        )

        predictions = self._predictions(
            packed_input,
            masked_idx,
            masked_positions,
            masked_counts,
            max_seqlen_q,
            cu_seqlens,
            max_seqlen,
        )

        return predictions, packed_targets, masked_idx
