"""Cache MiniLM prompt embeddings for reasoning datasets (see download_reasoning.py)."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

MODELS = ["r1-qwen7b", "r1-qwen14b", "qwen3-8b", "qwen3-14b"]
DSETS = ["aime25", "hmmtfeb25", "gpqa-diamond", "livecodebench-subset-v6"]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DATA_PKL = "r1_reasoning.pkl"
DEFAULT_EMBED_PKL = "r1_reasoning_embeddings.pkl"


def _cfg_name(ds: str, model: str) -> str:
    return f"{ds}_{model}"


def _prompts_to_text(prompts) -> list[str]:
    out = []
    for p in prompts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, (list, tuple)):
            out.append("\n".join(str(x) for x in p))
        else:
            out.append(str(p))
    return out


def load_prompts_from_data_pkl(path: Path) -> dict[str, list[str]]:
    with path.open("rb") as f:
        data_dict = pickle.load(f)
    return {
        cfg: _prompts_to_text(entry["prompt"])
        for cfg, entry in data_dict.items()
    }


def load_prompts_from_hub(models: list[str], dsets: list[str]) -> dict[str, list[str]]:
    import datasets

    prompts_by_cfg: dict[str, list[str]] = {}
    for model in models:
        for ds in dsets:
            cfg = _cfg_name(ds, model)
            start = perf_counter()
            dataset = datasets.load_dataset(
                f"drproduck/{model}-{ds}-n128",
                split="train",
                columns=["prompt"],
            )
            prompts_by_cfg[cfg] = _prompts_to_text(dataset["prompt"])
            print(f"{cfg}: {len(prompts_by_cfg[cfg])} prompts ({perf_counter() - start:.1f}s)")
    return prompts_by_cfg


def embed_prompts(
    prompts_by_cfg: dict[str, list[str]],
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str | None = None,
) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embed_dim = model.get_embedding_dimension()

    embeddings_by_cfg: dict[str, np.ndarray] = {}
    for cfg, prompts in prompts_by_cfg.items():
        start = perf_counter()
        emb = model.encode(
            prompts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape != (len(prompts), embed_dim):
            raise ValueError(f"{cfg}: expected ({len(prompts)}, {embed_dim}), got {emb.shape}")
        embeddings_by_cfg[cfg] = emb
        print(f"{cfg}: embeddings {emb.shape} ({perf_counter() - start:.1f}s)")

    return {
        "model_name": model_name,
        "embed_dim": int(embed_dim),
        "embeddings": embeddings_by_cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-pkl",
        type=Path,
        default=Path(DEFAULT_DATA_PKL),
        help="Pickle from download_reasoning.py (must include 'prompt' per config).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_EMBED_PKL),
        help="Output pickle path for cached embeddings.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence-Transformers model id.",
    )
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Download prompts from Hugging Face instead of --data-pkl.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="e.g. cuda, cpu (default: auto)")
    args = parser.parse_args()

    if args.from_hub:
        prompts_by_cfg = load_prompts_from_hub(MODELS, DSETS)
    else:
        if not args.data_pkl.is_file():
            raise FileNotFoundError(
                f"{args.data_pkl} not found. Run download_reasoning.py first or pass --from-hub."
            )
        prompts_by_cfg = load_prompts_from_data_pkl(args.data_pkl)

    payload = embed_prompts(
        prompts_by_cfg,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    with args.output.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Wrote {args.output} ({len(payload['embeddings'])} configs, dim={payload['embed_dim']})")


if __name__ == "__main__":
    main()
