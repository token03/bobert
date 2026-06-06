from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from core.data.batch import pack_batch
from core.data.beatmap import MAP_FEATURE_ATTRIBUTES
from core.data.normalizer import BeatmapNormalizer
from scripts.common.paths import resolve_path
from scripts.common.query import (
    DEFAULT_BEATMAPS_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_METADATA_PATH,
    beatmap_inputs_from_osu,
    ensure_osu_file,
    extract_beatmap_id,
    get_query_set_id,
    load_embeddings,
    load_metadata,
    metadata_by_id,
)
from scripts.tasks.embed.bobert import (
    load_alignment_model,
    normalize_checkpoint_state,
)
from scripts.tasks.query.recommend import (
    add_map_columns,
    map_cells,
    metadata_set_id,
    refresh_missing_metadata,
)

console = Console()
DEFAULT_MEMBERS = (
    (
        "v3.1",
        Path("experiments/align/checkpoints/v3.1.ckpt"),
        Path("data/embeddings-v1.parquet"),
    ),
    (
        "v3.2",
        Path("experiments/align/checkpoints/v3.2.ckpt"),
        Path("data/embeddings-v2.parquet"),
    ),
    (
        "v3.3",
        Path("experiments/align/checkpoints/v3.3.ckpt"),
        Path("data/embeddings-v3.parquet"),
    ),
)


@dataclass
class EnsembleMember:
    label: str
    checkpoint_path: Path
    embeddings_path: Path
    model: torch.nn.Module
    normalizer: BeatmapNormalizer


@dataclass
class EnsembleContext:
    beatmap_ids: np.ndarray
    head_embeddings: list[np.ndarray]
    centroid_embeddings: np.ndarray
    id_to_index: dict[int, int]
    metadata_lookup: dict[int, dict]
    metadata_path: Path
    beatmaps_dir: Path
    members: list[EnsembleMember]
    device: torch.device
    top_k: int
    candidate_k: int
    lambda_std: float
    include_same_set: bool
    allow_download: bool
    show_head_scores: bool
    max_seq_len: int
    shared_encoder: str
    warned_normalizer_mismatch: bool = False
    cache: dict[int, np.ndarray] = field(default_factory=dict)


def patch_config_from_state(config, state: dict[str, torch.Tensor]):
    query = state.get("contrastive_pooler.query")
    if query is not None:
        config.alignment.query_pool_num_queries = int(query.shape[0])
        config.alignment.query_pool_heads = int(query.shape[1])
        config.alignment.query_pool_head_dim = int(query.shape[2])

    out = state.get("contrastive_pooler.out.1.weight")
    if out is not None:
        config.alignment.query_pool_output_dim = int(out.shape[0])

    retrieval = state.get("retrieval_head.3.weight")
    if retrieval is not None:
        config.alignment.embedding_dim = int(retrieval.shape[0])


def load_checkpoint_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint, normalize_checkpoint_state(checkpoint.get("state_dict", checkpoint))


def verify_shared_encoder(states: list[dict[str, torch.Tensor]], labels: list[str]):
    base = states[0]
    keys = [key for key in base if key.startswith("bert.")]
    for label, state in zip(labels[1:], states[1:]):
        for key in keys:
            other = state.get(key)
            value = base[key]
            if other is None or tuple(other.shape) != tuple(value.shape):
                raise ValueError(f"{label} has incompatible encoder tensor: {key}")
            if not torch.equal(value, other):
                raise ValueError(f"{label} encoder differs at tensor: {key}")


def normalized_matrix(path: Path):
    beatmap_ids, embeddings, id_to_index = load_embeddings(path)
    return beatmap_ids, embeddings.astype(np.float16), id_to_index


def load_ensemble_embeddings(paths: list[Path]):
    beatmap_ids = None
    id_to_index = None
    head_embeddings = []
    centroid_sum = None

    for path in paths:
        ids, embeddings, lookup = normalized_matrix(path)
        if beatmap_ids is None:
            beatmap_ids = ids
            id_to_index = lookup
            centroid_sum = embeddings.astype(np.float32)
        else:
            if not np.array_equal(beatmap_ids, ids):
                raise ValueError(f"Embeddings IDs/order differ in {path}")
            centroid_sum += embeddings.astype(np.float32)
        head_embeddings.append(embeddings)
        console.print(
            f"[green]Loaded[/green] {len(ids):,} embeddings from "
            f"[dim]{escape(str(path))}[/dim]"
        )

    centroid = centroid_sum / len(head_embeddings)
    centroid /= np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-12)
    return beatmap_ids, head_embeddings, centroid.astype(np.float32), id_to_index


def load_members(args: argparse.Namespace):
    checkpoint_paths = [resolve_path(path) for _label, path, _emb in DEFAULT_MEMBERS]
    labels = [label for label, _path, _emb in DEFAULT_MEMBERS]
    checkpoints_and_states = [load_checkpoint_state(path) for path in checkpoint_paths]
    states = [state for _checkpoint, state in checkpoints_and_states]
    verify_shared_encoder(states, labels)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    members = []
    shared_bert = None
    for (label, _path, emb_path), checkpoint_path, (_checkpoint, state) in zip(
        DEFAULT_MEMBERS, checkpoint_paths, checkpoints_and_states
    ):
        config = OmegaConf.load(resolve_path(args.config))
        patch_config_from_state(config, state)
        model, loaded_checkpoint = load_alignment_model(config, checkpoint_path, device)
        if shared_bert is None:
            shared_bert = model.bert
        else:
            model.bert = shared_bert
        normalizer = BeatmapNormalizer(
            vector_stats=loaded_checkpoint["vector_stats"],
            attribute_stats=loaded_checkpoint.get("attribute_stats", {}),
        )
        members.append(
            EnsembleMember(
                label=label,
                checkpoint_path=checkpoint_path,
                embeddings_path=resolve_path(emb_path),
                model=model.eval(),
                normalizer=normalizer,
            )
        )
    return members, device


def prepare_query_inputs(
    vectors: np.ndarray,
    raw_map_features: dict[str, float],
    ctx: EnsembleContext,
):
    normalized_vectors = []
    normalized_map_features = []
    for member in ctx.members:
        normalized_vectors.append(member.normalizer.normalize_vectors(vectors))
        normalized_map_features.append(
            np.array(
                [
                    member.normalizer.normalize_attribute(
                        name, raw_map_features.get(name, 0.0)
                    )
                    for name in MAP_FEATURE_ATTRIBUTES
                ],
                dtype=np.float32,
            )
        )

    shared_inputs_valid = True
    for label, member_vectors, member_features in zip(
        [member.label for member in ctx.members[1:]],
        normalized_vectors[1:],
        normalized_map_features[1:],
    ):
        if not np.allclose(normalized_vectors[0], member_vectors, rtol=1e-5, atol=1e-6):
            shared_inputs_valid = False
            break
        if not np.allclose(
            normalized_map_features[0], member_features, rtol=1e-5, atol=1e-6
        ):
            shared_inputs_valid = False
            break

    if not shared_inputs_valid and ctx.shared_encoder == "require":
        raise ValueError(
            f"{label} normalizer produces different inputs; shared encoder pass is invalid"
        )

    return list(zip(normalized_vectors, normalized_map_features)), shared_inputs_valid


def run_head(
    model: torch.nn.Module,
    packed_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    map_features: torch.Tensor,
):
    contrastive_pooled = model.contrastive_pooler(
        packed_output,
        cu_seqlens,
        max_seqlen=max_seqlen,
    )
    aux_pooled = model.pooler(
        packed_output,
        cu_seqlens,
        max_seqlen=max_seqlen,
    )
    if model.map_projector is not None:
        map_projected = model._project_map_features(map_features, contrastive_pooled)
        pooled = torch.cat([contrastive_pooled, aux_pooled, map_projected], dim=-1)
    else:
        pooled = torch.cat([contrastive_pooled, aux_pooled], dim=-1)
    return F.normalize(model.retrieval_head(pooled), dim=-1)


def embed_osu(path: Path, ctx: EnsembleContext):
    vectors, raw_map_features = beatmap_inputs_from_osu(path, ctx.max_seq_len)
    query_inputs, shared_inputs_valid = prepare_query_inputs(
        vectors, raw_map_features, ctx
    )
    amp_dtype = torch.bfloat16 if ctx.device.type == "cuda" else torch.float32
    shared_bert = ctx.members[0].model.bert

    with torch.inference_mode():
        with torch.autocast(
            device_type=ctx.device.type,
            dtype=amp_dtype,
            enabled=ctx.device.type == "cuda",
        ):
            if shared_inputs_valid and ctx.shared_encoder != "off":
                normalized_vectors, map_features_np = query_inputs[0]
                packed, cu_seqlens, max_seqlen = pack_batch(
                    [normalized_vectors], ctx.max_seq_len, normalized_vectors.shape[1]
                )
                map_features = torch.tensor(
                    map_features_np, dtype=torch.float32
                ).unsqueeze(0)
                packed = packed.to(ctx.device)
                cu_seqlens = cu_seqlens.to(ctx.device)
                map_features = map_features.to(ctx.device)
                packed_input = shared_bert.embed_sequences(packed)
                packed_output = shared_bert.encode(
                    packed_input,
                    attention_mask=None,
                    max_seqlen=max_seqlen,
                    cu_seqlens=cu_seqlens,
                )
                embeddings = [
                    run_head(
                        member.model, packed_output, cu_seqlens, max_seqlen, map_features
                    )
                    for member in ctx.members
                ]
            else:
                if not shared_inputs_valid and not ctx.warned_normalizer_mismatch:
                    console.print(
                        "[yellow]Warning:[/yellow] checkpoint normalizers produce "
                        "different query inputs; using one encoder pass per head."
                    )
                    ctx.warned_normalizer_mismatch = True
                embeddings = []
                for member, (normalized_vectors, map_features_np) in zip(
                    ctx.members, query_inputs
                ):
                    packed, cu_seqlens, max_seqlen = pack_batch(
                        [normalized_vectors], ctx.max_seq_len, normalized_vectors.shape[1]
                    )
                    map_features = torch.tensor(
                        map_features_np, dtype=torch.float32
                    ).unsqueeze(0)
                    packed = packed.to(ctx.device)
                    cu_seqlens = cu_seqlens.to(ctx.device)
                    map_features = map_features.to(ctx.device)
                    packed_input = shared_bert.embed_sequences(packed)
                    packed_output = shared_bert.encode(
                        packed_input,
                        attention_mask=None,
                        max_seqlen=max_seqlen,
                        cu_seqlens=cu_seqlens,
                    )
                    embeddings.append(
                        run_head(
                            member.model,
                            packed_output,
                            cu_seqlens,
                            max_seqlen,
                            map_features,
                        )
                    )

    head_embeddings = torch.cat(embeddings, dim=0).float().cpu().numpy()
    head_embeddings /= np.maximum(
        np.linalg.norm(head_embeddings, axis=1, keepdims=True), 1e-12
    )
    return head_embeddings.astype(np.float32)


def get_head_embeddings(raw_input: str, ctx: EnsembleContext):
    beatmap_id = extract_beatmap_id(raw_input)
    if beatmap_id in ctx.cache:
        return beatmap_id, ctx.cache[beatmap_id]

    osu_path = ensure_osu_file(beatmap_id, ctx.beatmaps_dir, ctx.allow_download)
    embeddings = embed_osu(osu_path, ctx)
    ctx.cache[beatmap_id] = embeddings
    return beatmap_id, embeddings


def centroid_embedding(head_embeddings: np.ndarray):
    centroid = head_embeddings.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    return centroid.astype(np.float32)


def score_candidates(
    query_heads: np.ndarray,
    candidate_indices: np.ndarray,
    ctx: EnsembleContext,
):
    scores = []
    for query, embeddings in zip(query_heads, ctx.head_embeddings):
        candidates = embeddings[candidate_indices].astype(np.float32)
        scores.append(candidates @ query.astype(np.float32))
    head_scores = np.stack(scores, axis=1)
    mean = head_scores.mean(axis=1)
    std = head_scores.std(axis=1)
    final = mean - ctx.lambda_std * std
    return final, mean, std, head_scores


def ranked_candidates(beatmap_id: int, query_heads: np.ndarray, ctx: EnsembleContext):
    query_centroid = centroid_embedding(query_heads)
    centroid_scores = ctx.centroid_embeddings @ query_centroid
    candidate_count = min(ctx.candidate_k, len(ctx.beatmap_ids))
    candidate_indices = np.argpartition(-centroid_scores, candidate_count - 1)[
        :candidate_count
    ]
    final, mean, std, head_scores = score_candidates(query_heads, candidate_indices, ctx)
    order = np.argsort(-final)
    for position in order:
        idx = int(candidate_indices[position])
        candidate_id = int(ctx.beatmap_ids[idx])
        if candidate_id == beatmap_id:
            continue
        yield {
            "beatmap_id": candidate_id,
            "final": float(final[position]),
            "mean": float(mean[position]),
            "std": float(std[position]),
            "heads": head_scores[position].astype(float).tolist(),
            "row": ctx.metadata_lookup.get(candidate_id),
        }


def print_query_table(beatmap_id: int, ctx: EnsembleContext, row: dict | None = None):
    table = Table(title="Query", show_header=True, header_style="bold magenta")
    add_map_columns(table)
    table.add_row(*map_cells(beatmap_id, row or ctx.metadata_lookup.get(beatmap_id)))
    console.print()
    console.print(table)


def print_result_table(results: list[dict], ctx: EnsembleContext):
    table = Table(title="Ensemble", show_header=True, header_style="bold magenta")
    table.add_column("Score", justify="right", style="green", no_wrap=True)
    if ctx.show_head_scores:
        table.add_column("Mean", justify="right", style="green", no_wrap=True)
        table.add_column("Std", justify="right", style="yellow", no_wrap=True)
        for member in ctx.members:
            table.add_column(member.label, justify="right", no_wrap=True)
    add_map_columns(table)
    for result in results:
        values = [f"{result['final']:.3f}"]
        if ctx.show_head_scores:
            values.extend(
                [
                    f"{result['mean']:.3f}",
                    f"{result['std']:.3f}",
                    *[f"{score:.3f}" for score in result["heads"]],
                ]
            )
        table.add_row(*values, *map_cells(result["beatmap_id"], result["row"]))
    console.print(table)


def recommend(raw_input: str, ctx: EnsembleContext):
    beatmap_id, query_heads = get_head_embeddings(raw_input, ctx)
    query_row = refresh_missing_metadata(
        [(beatmap_id, 0.0, ctx.metadata_lookup.get(beatmap_id))], ctx
    )[0][2]
    query_set_id = get_query_set_id(beatmap_id, raw_input, ctx.metadata_lookup)

    results = []
    seen_set_ids = set()
    for result in ranked_candidates(beatmap_id, query_heads, ctx):
        candidate_set_id = metadata_set_id(result["row"])
        if (
            not ctx.include_same_set
            and query_set_id is not None
            and candidate_set_id == query_set_id
        ):
            continue
        if candidate_set_id is not None and candidate_set_id in seen_set_ids:
            continue

        results.append(result)
        if candidate_set_id is not None:
            seen_set_ids.add(candidate_set_id)
        if len(results) >= ctx.top_k:
            break

    print_query_table(beatmap_id, ctx, query_row)
    if query_set_id is not None and not ctx.include_same_set:
        console.print(f"[dim]Excluding same beatmapset: {query_set_id}[/dim]")
    refreshed = refresh_missing_metadata(
        [(result["beatmap_id"], result["final"], result["row"]) for result in results],
        ctx,
    )
    for result, (_beatmap_id, _score, row) in zip(results, refreshed):
        result["row"] = row
    print_result_table(results, ctx)
    console.print()


def compare(raw_input_a: str, raw_input_b: str, ctx: EnsembleContext):
    beatmap_id_a, heads_a = get_head_embeddings(raw_input_a, ctx)
    beatmap_id_b, heads_b = get_head_embeddings(raw_input_b, ctx)
    scores = np.sum(heads_a * heads_b, axis=1)
    mean = float(scores.mean())
    std = float(scores.std())
    final = mean - ctx.lambda_std * std

    table = Table(show_header=True, header_style="bold magenta")
    add_map_columns(table, side=True)
    table.add_row("A", *map_cells(beatmap_id_a, ctx.metadata_lookup.get(beatmap_id_a)))
    table.add_row("B", *map_cells(beatmap_id_b, ctx.metadata_lookup.get(beatmap_id_b)))

    console.print()
    console.print(table)
    console.print(
        f"[bold]Similarity:[/bold] [green]{final:.6f}[/green] "
        f"[dim](mean={mean:.6f}, std={std:.6f})[/dim]\n"
    )


def run_query(parts: list[str], ctx: EnsembleContext):
    if len(parts) == 1:
        recommend(parts[0], ctx)
    elif len(parts) == 2:
        compare(parts[0], parts[1], ctx)
    else:
        raise ValueError("enter one beatmap for recommendations or two for comparison")


def run_interactive(ctx: EnsembleContext):
    console.print(
        "[dim]Paste one beatmap id/URL for recommendations, or two for comparison.[/dim]"
    )
    console.print(
        "[dim]Press Ctrl+C to clear, or Ctrl+D, q, quit, or empty input to exit.[/dim]"
    )
    while True:
        try:
            raw_input = console.input("[bold cyan]query>[/bold cyan] ").strip()
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print()
            return

        if "\x03" in raw_input:
            console.print()
            continue

        if raw_input.lower() in {"", "q", "quit", "exit"}:
            return

        try:
            run_query(raw_input.split(), ctx)
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend similar beatmaps with a three-head Bobert ensemble"
    )
    parser.add_argument(
        "beatmaps",
        nargs="*",
        help="One beatmap id/URL recommends; two beatmap ids/URLs compares.",
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--beatmaps-dir", default=str(DEFAULT_BEATMAPS_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--lambda-std", type=float, default=0.5)
    parser.add_argument("--include-same-set", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--show-head-scores", action="store_true")
    parser.add_argument(
        "--shared-encoder",
        default="auto",
        choices=("auto", "require", "off"),
        help="Use one encoder pass when normalized inputs match; require errors on mismatch.",
    )
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    if len(args.beatmaps) > 2:
        raise SystemExit("Error: provide at most two beatmap ids or URLs")
    if args.top_k <= 0:
        raise SystemExit("Error: --top-k must be positive")
    if args.candidate_k is not None and args.candidate_k <= 0:
        raise SystemExit("Error: --candidate-k must be positive")


def build_context(args: argparse.Namespace):
    members, device = load_members(args)
    beatmap_ids, head_embeddings, centroid_embeddings, id_to_index = load_ensemble_embeddings(
        [member.embeddings_path for member in members]
    )
    candidate_k = args.candidate_k or max(args.top_k * 10, 200)
    return EnsembleContext(
        beatmap_ids=beatmap_ids,
        head_embeddings=head_embeddings,
        centroid_embeddings=centroid_embeddings,
        id_to_index=id_to_index,
        metadata_lookup=metadata_by_id(load_metadata(resolve_path(args.metadata))),
        metadata_path=resolve_path(args.metadata),
        beatmaps_dir=resolve_path(args.beatmaps_dir),
        members=members,
        device=device,
        top_k=args.top_k,
        candidate_k=candidate_k,
        lambda_std=args.lambda_std,
        include_same_set=args.include_same_set,
        allow_download=not args.no_download,
        show_head_scores=args.show_head_scores,
        max_seq_len=int(OmegaConf.load(resolve_path(args.config)).data.max_seq_len),
        shared_encoder=args.shared_encoder,
    )


def main():
    args = parse_args()
    validate_args(args)
    ctx = build_context(args)
    run_query(args.beatmaps, ctx) if args.beatmaps else run_interactive(ctx)


if __name__ == "__main__":
    main()
