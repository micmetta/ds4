#!/usr/bin/env python3
"""Extract a compact frozen token-embedding lookup from one GGUF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from gguf import GGUFReader


DEFAULT_TENSOR_NAME = "token_embd.weight"
DEFAULT_BATCH_SIZE = 100_000


def parse_arguments() -> argparse.Namespace:
    """Read the Parquet, GGUF and output paths from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Estrae dal GGUF gli embedding congelati dei token presenti "
            "in un Parquet di training."
        )
    )
    parser.add_argument("--training-parquet", type=Path, required=True)
    parser.add_argument("--gguf-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR_NAME)
    return parser.parse_args()


def require_new_output_directory(output_dir: Path) -> None:
    """Refuse an existing destination to preserve prior experiment artifacts."""
    if output_dir.exists():
        raise FileExistsError(
            "La directory di output esiste gia': "
            f"{output_dir}. Scegli una nuova directory per non sovrascrivere artefatti."
        )


def collect_token_ids(training_parquet: Path) -> np.ndarray:
    """Return sorted unique non-negative token IDs using streaming Parquet reads."""
    if not training_parquet.is_file():
        raise FileNotFoundError(f"Parquet non trovato: {training_parquet}")

    parquet = pq.ParquetFile(training_parquet)
    if "token_id" not in parquet.schema_arrow.names:
        raise ValueError("Il Parquet deve contenere la colonna token_id.")

    token_ids: set[int] = set()
    for batch in parquet.iter_batches(
        columns=["token_id"], batch_size=DEFAULT_BATCH_SIZE
    ):
        values = batch.column("token_id").to_numpy(zero_copy_only=False)
        if np.any(np.isnan(values)):
            raise ValueError("La colonna token_id contiene valori nulli o NaN.")
        integer_values = np.asarray(values, dtype=np.int64)
        if np.any(integer_values < 0):
            raise ValueError("token_id deve contenere soltanto valori non negativi.")
        token_ids.update(map(int, integer_values))

    if not token_ids:
        raise ValueError("Il Parquet non contiene alcun token_id.")
    return np.asarray(sorted(token_ids), dtype=np.int32)


def find_tensor(reader: GGUFReader, tensor_name: str):
    """Find one named GGUF tensor without materializing unrelated weights."""
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            return tensor
    available = [tensor.name for tensor in reader.tensors if "embd" in tensor.name]
    raise ValueError(
        f"Tensore GGUF non trovato: {tensor_name}. "
        f"Tensori embedding disponibili: {available}"
    )


def save_json(path: Path, content: dict[str, object]) -> None:
    """Write UTF-8 JSON with stable readable formatting."""
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Extract and save the frozen compact lookup and its reproducibility metadata."""
    args = parse_arguments()
    require_new_output_directory(args.output_dir)
    if not args.gguf_model.is_file():
        raise FileNotFoundError(f"Modello GGUF non trovato: {args.gguf_model}")

    print("1) Leggo gli ID token distinti dal Parquet in streaming...")
    token_ids = collect_token_ids(args.training_parquet)
    print(f"   token ID distinti: {len(token_ids):,}")
    print(f"   intervallo osservato: [{token_ids[0]}, {token_ids[-1]}]")

    print("2) Apro il GGUF e verifico il tensore degli embedding token...")
    reader = GGUFReader(args.gguf_model, mode="r")
    tensor = find_tensor(reader, args.tensor_name)
    source_embeddings = tensor.data
    if source_embeddings.ndim != 2:
        raise ValueError(
            f"{args.tensor_name}: attesa matrice 2D, trovata shape {source_embeddings.shape}."
        )
    vocabulary_size, embedding_dimension = source_embeddings.shape
    if token_ids[-1] >= vocabulary_size:
        raise ValueError(
            "Il Parquet contiene token_id fuori dal vocabolario GGUF: "
            f"massimo dataset={token_ids[-1]}, vocab_size={vocabulary_size}."
        )
    print(
        f"   {args.tensor_name}: vocab={vocabulary_size:,}, "
        f"dim={embedding_dimension:,}, dtype={source_embeddings.dtype}"
    )

    print("3) Estraggo soltanto gli embedding token presenti nel build...")
    selected_embeddings = np.ascontiguousarray(source_embeddings[token_ids])
    if selected_embeddings.shape != (len(token_ids), embedding_dimension):
        raise ValueError("La shape del lookup estratto non e' coerente.")

    args.output_dir.mkdir(parents=True)
    token_ids_path = args.output_dir / "token_ids.npy"
    embeddings_path = args.output_dir / "token_embeddings_f16.npy"
    summary_path = args.output_dir / "token_embedding_lookup_summary.json"
    np.save(token_ids_path, token_ids)
    np.save(embeddings_path, selected_embeddings.astype(np.float16, copy=False))
    save_json(
        summary_path,
        {
            "experiment": "frozen_gguf_token_embedding_lookup",
            "training_parquet": str(args.training_parquet),
            "gguf_model": str(args.gguf_model),
            "gguf_model_size_bytes": args.gguf_model.stat().st_size,
            "tensor_name": args.tensor_name,
            "tensor_dtype": str(source_embeddings.dtype),
            "vocabulary_size": int(vocabulary_size),
            "embedding_dimension": int(embedding_dimension),
            "selected_token_count": int(len(token_ids)),
            "selected_token_id_min": int(token_ids[0]),
            "selected_token_id_max": int(token_ids[-1]),
            "token_ids_path": str(token_ids_path),
            "embeddings_path": str(embeddings_path),
            "lookup_mapping": {
                "raw_token_id_storage": "token_ids.npy sorted ascending",
                "embedding_row_for_raw_token_id": "numpy.searchsorted(token_ids, raw_token_id)",
                "contains_only_ids_observed_in_training_parquet": True,
            },
            "embedding_usage": {
                "frozen": True,
                "future_trainable_projection": "Dense(4096 -> 64)",
                "fit_uses_labels": False,
            },
        },
    )

    print("\nLookup token embedding completato.")
    print(f"Token IDs: {token_ids_path}")
    print(f"Embedding: {embeddings_path}")
    print(f"Report: {summary_path}")


if __name__ == "__main__":
    main()
