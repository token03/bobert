from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import artifacts

DENSITY_K = 500
LAMBDA = 0.6
DENSITY_POWER = 3.0
SCORE_CHUNK = 2048
SCORE_WORKERS = 2

_score_pool = ThreadPoolExecutor(SCORE_WORKERS)


def density_term(
    densities: np.ndarray | torch.Tensor, lambda_: float = LAMBDA
) -> np.ndarray | torch.Tensor:
    return 0.5 * lambda_ * densities


def bfloat16_scores(
    bits: np.ndarray, query: np.ndarray, rows: np.ndarray | None = None
) -> np.ndarray:
    count = len(bits) if rows is None else len(rows)
    query = np.asarray(query, dtype=np.float32)
    scores = np.empty(count, dtype=np.float32)

    def score(start: int, stop: int) -> None:
        wide = np.empty((SCORE_CHUNK, bits.shape[1]), dtype=np.uint32)
        for begin in range(start, stop, SCORE_CHUNK):
            end = min(begin + SCORE_CHUNK, stop)
            block = bits[begin:end] if rows is None else bits[rows[begin:end]]
            values = wide[: end - begin]
            np.left_shift(block, 16, out=values, dtype=np.uint32)
            np.matmul(values.view(np.float32), query, out=scores[begin:end])

    bounds = np.linspace(0, count, SCORE_WORKERS + 1).astype(int)
    list(_score_pool.map(score, bounds[:-1], bounds[1:]))
    return scores


def compute_densities(
    embeddings: np.ndarray,
    device_name: str | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    if len(embeddings) <= DENSITY_K:
        raise ValueError(f"need more than {DENSITY_K} embeddings to index retrieval")
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    corpus = torch.from_numpy(embeddings).to(device=device, dtype=dtype)
    densities = np.empty(len(embeddings), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(corpus), batch_size):
            stop = min(start + batch_size, len(corpus))
            scores = corpus[start:stop] @ corpus.T
            scores[
                torch.arange(stop - start, device=device),
                torch.arange(start, stop, device=device),
            ] = -torch.inf
            densities[start:stop] = (
                scores.topk(DENSITY_K, dim=1, sorted=False)
                .values.float()
                .mean(dim=1)
                .cpu()
                .numpy()
            )

    deviation = densities.std()
    transformed = densities**DENSITY_POWER
    return (
        (transformed - transformed.mean()) / transformed.std() * deviation
        + densities.mean()
    ).astype(np.float32)


def index_retrieval(
    path: str,
    model: str,
    device_name: str | None = None,
    batch_size: int = 1024,
    quiet: bool = False,
) -> None:
    index = artifacts.read_index(path, with_density=False)
    if not quiet:
        print(f"Indexing retrieval for {len(index.embeddings):,} embeddings")
    densities = compute_densities(index.vectors(), device_name, batch_size)
    metadata = {
        **index.metadata,
        "retrieval": {
            "method": "csls",
            "density_k": DENSITY_K,
            "lambda": LAMBDA,
            "density_power": DENSITY_POWER,
        },
    }
    artifacts.write_index(
        path,
        index.ids,
        index.embeddings,
        artifacts.from_bfloat16,
        metadata,
        model=model,
        batch_size=batch_size,
        densities=densities,
        quiet=True,
    )
