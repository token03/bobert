from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

import torch

from core.components import EncoderLayer
from core.dataset import Packed
from core.model import BobertEncoder


def _layers(model: BobertEncoder) -> list[EncoderLayer]:
    return [cast(EncoderLayer, layer) for layer in model.layers]


def _top(scores: torch.Tensor, count: int) -> torch.Tensor:
    return torch.topk(scores, count).indices.sort().values


@torch.no_grad()
def calibrate_importance(
    teacher: BobertEncoder, batches: Iterable[Packed]
) -> dict[str, torch.Tensor]:
    device = teacher.layer_means.device
    layers = _layers(teacher)
    head_dim = teacher.d_model // teacher.n_heads
    feedforward = layers[0].ffn.w2.in_features
    embedding = torch.zeros(teacher.d_model, dtype=torch.float64, device=device)
    heads = torch.zeros(
        len(layers), teacher.n_heads, dtype=torch.float64, device=device
    )
    ffn = torch.zeros(len(layers), feedforward, dtype=torch.float64, device=device)

    def on_norm(_module, _inputs, output):
        embedding.add_(output.float().square().sum(dim=0))

    def on_heads(index: int):
        def hook(_module, inputs, _output):
            values = inputs[0].float().view(-1, teacher.n_heads, head_dim)
            heads[index].add_(values.square().sum(dim=(0, 2)))

        return hook

    def on_ffn(index: int):
        def hook(_module, inputs, _output):
            ffn[index].add_(inputs[0].float().square().sum(dim=0))

        return hook

    hooks = [teacher.final_norm.register_forward_hook(on_norm)]
    for index, layer in enumerate(layers):
        hooks += [
            layer.norm1.register_forward_hook(on_norm),
            layer.norm2.register_forward_hook(on_norm),
            layer.self_attn.wo.register_forward_hook(on_heads(index)),
            layer.ffn.w2.register_forward_hook(on_ffn(index)),
        ]
    for batch in batches:
        teacher.embed(batch.to(device))
    for hook in hooks:
        hook.remove()

    return {
        "emb": embedding.sqrt().float().cpu(),
        "head": heads.sqrt().float().cpu(),
        "ffn": ffn.sqrt().float().cpu(),
    }


def _slice_layer(
    student: EncoderLayer,
    teacher: EncoderLayer,
    keep_dims: torch.Tensor,
    keep_heads: torch.Tensor,
    keep_ffn: torch.Tensor,
    heads: int,
) -> None:
    def weight(module: torch.nn.Module) -> torch.Tensor:
        return cast(torch.Tensor, module.weight).detach().float().cpu()

    width = len(keep_dims)
    head_dim = teacher.self_attn.wo.out_features // heads
    qkv = [
        part.view(heads, head_dim, -1)[keep_heads][:, :, keep_dims].reshape(-1, width)
        for part in weight(teacher.self_attn.wqkv).chunk(3)
    ]
    output = weight(teacher.self_attn.wo)[keep_dims].view(width, heads, head_dim)
    gate, up = weight(teacher.ffn.w13).chunk(2)

    student.self_attn.wqkv.weight.copy_(torch.cat(qkv))
    student.self_attn.wo.weight.copy_(output[:, keep_heads].reshape(width, -1))
    student.ffn.w13.weight.copy_(
        torch.cat([gate[keep_ffn][:, keep_dims], up[keep_ffn][:, keep_dims]])
    )
    student.ffn.w2.weight.copy_(weight(teacher.ffn.w2)[keep_dims][:, keep_ffn])
    student.norm1.weight.copy_(weight(teacher.norm1)[keep_dims])
    student.norm2.weight.copy_(weight(teacher.norm2)[keep_dims])


@torch.no_grad()
def init_pruned_student(
    student: BobertEncoder,
    teacher: BobertEncoder,
    importance: dict[str, torch.Tensor],
    keep_layers: Sequence[int],
    spectral_components: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    keep_dims = _top(importance["emb"], student.d_model)
    student_tokenizer, teacher_tokenizer = student.tokenizer, teacher.tokenizer
    student_tokenizer.numeric_weight.copy_(teacher_tokenizer.numeric_weight)
    student_tokenizer.numeric_bias.copy_(teacher_tokenizer.numeric_bias)
    student_tokenizer.categorical_weight.copy_(teacher_tokenizer.categorical_weight)
    cast(torch.Tensor, student_tokenizer.normalization).copy_(
        cast(torch.Tensor, teacher_tokenizer.normalization)
    )
    student_tokenizer.out.weight.copy_(teacher_tokenizer.out.weight[keep_dims])
    student_tokenizer.norm.weight.copy_(teacher_tokenizer.norm.weight[keep_dims])

    student_feedforward = _layers(student)[0].ffn.w2.in_features
    teacher_layers = _layers(teacher)
    for student_layer, teacher_index in zip(_layers(student), keep_layers, strict=True):
        _slice_layer(
            student_layer,
            teacher_layers[teacher_index],
            keep_dims,
            _top(importance["head"][teacher_index], student.n_heads),
            _top(importance["ffn"][teacher_index], student_feedforward),
            teacher.n_heads,
        )
    student.final_norm.weight.copy_(teacher.final_norm.weight[keep_dims])

    teacher_globals = sorted(teacher.global_attention_layers)
    teacher_taps = [
        teacher_globals.index(keep_layers[layer])
        for layer in sorted(student.global_attention_layers)
    ]
    student.layer_means.copy_(teacher.layer_means[teacher_taps][:, keep_dims])
    adapter = spectral_components.T.cpu() @ teacher.adapter.proj.weight.float().cpu()
    student.adapter.proj.weight.copy_(adapter[:, keep_dims])
    return keep_dims, teacher_taps
