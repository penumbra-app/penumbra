"""Cache frozen, local text embeddings for controlled content experiments.

Only model downloads use the network; movie text is encoded on this machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from src.content.text import movie_text
from src.data_processing.movielens import movie_metadata_from_records, user_ratings_from_records
from src.evaluate_content import prepare_cases, read_csv


MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
}
REVISIONS = {
    "minilm": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "qwen": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
}
MAX_LENGTHS = {"minilm": 256, "qwen": 512}


def document_hash(documents):
    return hashlib.sha256(json.dumps(documents, ensure_ascii=False).encode()).hexdigest()


def experiment_documents(movies, cases):
    # No rating values influence text or embeddings. Include candidates only to
    # precompute frozen transforms; TF-IDF fitting uses the base catalog separately.
    return sorted({movie_text(m, "combined") for m in movies}
                  | {movie_text(m, "combined") for c in cases for m in c.movies})


def encode(name, documents, output, device=None):
    os.environ.setdefault("HF_HOME", str(Path(".cache/content-experiments/hf").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer
    import torch

    torch.manual_seed(42)
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / name
    digest = document_hash(documents)
    if prefix.with_suffix(".json").exists() and prefix.with_suffix(".npy").exists():
        metadata = json.loads(prefix.with_suffix(".json").read_text())
        if (metadata["documents_sha256"] == digest and metadata["model"] == MODELS[name]
                and metadata["revision"] == REVISIONS[name]
                and metadata["max_seq_length"] == MAX_LENGTHS[name]
                and metadata["matrix_sha256"] == hashlib.sha256(prefix.with_suffix(".npy").read_bytes()).hexdigest()):
            print(f"{name}: using verified embedding cache", flush=True)
            return
    started = time.perf_counter()
    revision = REVISIONS[name]
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{name}: loading {MODELS[name]} revision={revision} device={device}", flush=True)
    model = SentenceTransformer(MODELS[name], revision=revision, device=device,
                               trust_remote_code=False, token=False)
    # Use each model's documented pooling; no query instruction for movie/movie matching.
    model.max_seq_length = MAX_LENGTHS[name]
    if name == "qwen" and device == "mps":
        model.half()
    model.eval()
    matrix = np.zeros((len(documents), model.get_sentence_embedding_dimension()), dtype=np.float32)
    nonempty = [i for i, text in enumerate(documents) if text.strip()]
    load_seconds = time.perf_counter() - started
    for offset in range(0, len(nonempty), 256):
        rows = nonempty[offset:offset + 256]
        matrix[rows] = model.encode([documents[i] for i in rows], batch_size=16 if name == "qwen" else 64,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False)
        print(f"{name}: {min(offset + 256, len(nonempty))}/{len(nonempty)} documents; "
              f"{time.perf_counter() - started:.1f}s", flush=True)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite embedding")
    np.save(prefix.with_suffix(".npy"), matrix)
    metadata = {"model": MODELS[name], "revision": revision, "documents_sha256": digest,
                "documents": len(documents), "dimension": matrix.shape[1], "device": device,
                "max_seq_length": model.max_seq_length, "normalized": True,
                "pooling": "model default; no query instruction", "frozen": True,
                "load_seconds": load_seconds, "elapsed_seconds": time.perf_counter() - started,
                "matrix_sha256": hashlib.sha256(prefix.with_suffix(".npy").read_bytes()).hexdigest()}
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--movies-file", default="data/movies_enriched.csv")
    parser.add_argument("--output", default=".cache/content-experiments")
    parser.add_argument("--device", choices=("cpu", "mps"))
    args = parser.parse_args()
    movies = movie_metadata_from_records(read_csv(Path(args.movies_file)))
    cases, _ = prepare_cases(user_ratings_from_records(read_csv(Path(args.data_dir) / "ratings.csv")),
                             movies, read_csv(Path(args.data_dir) / "tags.csv"))
    documents = experiment_documents(movies, cases)
    for name in args.models:
        encode(name, documents, Path(args.output), args.device)
