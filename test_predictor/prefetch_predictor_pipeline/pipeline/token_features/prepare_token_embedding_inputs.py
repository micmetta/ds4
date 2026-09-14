#!/usr/bin/env python3
"""Prepare compact token inputs for the frozen-GGUF-embedding FFN ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SPLITS = ("train", "validation", "test")
BATCH_SIZE = 100_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara indici embedding token e token_position per una FFN multi-input."
    )
    parser.add_argument("--training-parquet", type=Path, required=True)
    parser.add_argument("--token-embedding-dir", type=Path, required=True)
    parser.add_argument("--state-arrays-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require_new_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"La directory di output esiste gia': {output_dir}")


def load_lookup(embedding_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    token_ids_path = embedding_dir / "token_ids.npy"
    embeddings_path = embedding_dir / "token_embeddings_f16.npy"
    if not token_ids_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError("Attesi token_ids.npy e token_embeddings_f16.npy.")
    token_ids = np.load(token_ids_path, mmap_mode="r")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if token_ids.ndim != 1 or embeddings.ndim != 2 or embeddings.shape[0] != len(token_ids):
        raise ValueError("Il lookup token embedding non ha shape coerenti.")
    if not np.all(token_ids[:-1] < token_ids[1:]):
        raise ValueError("token_ids.npy deve essere strettamente ordinato.")
    return token_ids, embeddings


def check_state_arrays(state_arrays_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        labels_path = state_arrays_dir / f"y_{split}.npy"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Label state non trovate: {labels_path}")
        counts[split] = int(np.load(labels_path, mmap_mode="r").shape[0])
    return counts


def main() -> None:
    args = parse_arguments()
    require_new_output_directory(args.output_dir)
    if not args.training_parquet.is_file():
        raise FileNotFoundError(f"Parquet non trovato: {args.training_parquet}")

    token_ids, embeddings = load_lookup(args.token_embedding_dir)
    expected_counts = check_state_arrays(args.state_arrays_dir)
    parquet = pq.ParquetFile(args.training_parquet)
    required = {"split", "token_id", "token_position"}
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError(f"Parquet privo delle colonne richieste: {sorted(required)}")

    print("1) Calcolo statistiche train di token_position e verifico il lookup...")
    counts = {split: 0 for split in SPLITS}
    train_sum = 0.0
    train_square_sum = 0.0
    for batch in parquet.iter_batches(columns=["split", "token_id", "token_position"], batch_size=BATCH_SIZE):
        splits = batch.column("split").to_pylist()
        raw_ids = np.asarray(batch.column("token_id").to_numpy(zero_copy_only=False), dtype=np.int64)
        positions = np.asarray(batch.column("token_position").to_numpy(zero_copy_only=False), dtype=np.float64)
        indices = np.searchsorted(token_ids, raw_ids)
        valid = (indices < len(token_ids)) & (token_ids[np.minimum(indices, len(token_ids) - 1)] == raw_ids)
        if not valid.all():
            missing = int((~valid).sum())
            raise ValueError(f"{missing} token_id non sono presenti nel lookup compatto.")
        for split in SPLITS:
            mask = np.fromiter((value == split for value in splits), dtype=bool, count=len(splits))
            counts[split] += int(mask.sum())
            if split == "train" and mask.any():
                selected = positions[mask]
                train_sum += float(selected.sum())
                train_square_sum += float(np.square(selected).sum())
    if counts != expected_counts:
        raise ValueError(f"Ordine/count Parquet incompatibile con state arrays: {counts} != {expected_counts}")
    train_mean = train_sum / counts["train"]
    train_variance = train_square_sum / counts["train"] - train_mean**2
    train_std = float(np.sqrt(max(train_variance, 0.0)))
    if train_std == 0:
        raise ValueError("token_position ha deviazione standard train nulla.")

    print("2) Scrivo indici compatti e token_position standardizzata in streaming...")
    arrays_dir = args.output_dir / "arrays"
    arrays_dir.mkdir(parents=True)
    index_outputs = {
        split: np.lib.format.open_memmap(
            arrays_dir / f"token_embedding_index_{split}.npy", mode="w+", dtype=np.int32,
            shape=(counts[split],)
        )
        for split in SPLITS
    }
    position_outputs = {
        split: np.lib.format.open_memmap(
            arrays_dir / f"token_position_{split}.npy", mode="w+", dtype=np.float32,
            shape=(counts[split], 1)
        )
        for split in SPLITS
    }
    offsets = {split: 0 for split in SPLITS}
    for batch in parquet.iter_batches(columns=["split", "token_id", "token_position"], batch_size=BATCH_SIZE):
        splits = np.asarray(batch.column("split").to_pylist(), dtype=object)
        raw_ids = np.asarray(batch.column("token_id").to_numpy(zero_copy_only=False), dtype=np.int64)
        positions = np.asarray(batch.column("token_position").to_numpy(zero_copy_only=False), dtype=np.float64)
        indices = np.searchsorted(token_ids, raw_ids).astype(np.int32, copy=False)
        for split in SPLITS:
            mask = splits == split
            size = int(mask.sum())
            if not size:
                continue
            start, stop = offsets[split], offsets[split] + size
            index_outputs[split][start:stop] = indices[mask]
            position_outputs[split][start:stop, 0] = ((positions[mask] - train_mean) / train_std).astype(np.float32)
            offsets[split] = stop
    if offsets != counts:
        raise RuntimeError("Scrittura incompleta degli input token.")

    summary = {
        "experiment": "token_embedding_ffn_inputs",
        "training_parquet": str(args.training_parquet),
        "state_arrays_dir": str(args.state_arrays_dir),
        "token_embedding_dir": str(args.token_embedding_dir),
        "embedding_table_shape": [int(value) for value in embeddings.shape],
        "token_embedding_index_dtype": "int32",
        "token_position": {
            "standardization": "(x - train_mean) / train_std",
            "train_mean": train_mean,
            "train_std": train_std,
            "fit_split": "train",
        },
        "split_examples": counts,
        "state_feature_source": "external sparse matrices; not copied into this ablation",
    }
    (args.output_dir / "artifacts").mkdir()
    (args.output_dir / "artifacts" / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\nInput token preparati.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
