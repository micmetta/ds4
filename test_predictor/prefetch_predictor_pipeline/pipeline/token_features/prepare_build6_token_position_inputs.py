#!/usr/bin/env python3
"""Materialize only the standardized token position required by Build 6.

Build 6 deliberately has no token-ID lookup and no frozen token embedding.
Keeping this small step separate makes that absence explicit and preserves the
same train-only standardization rule used by Build 5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SPLITS = ("train", "validation", "test")
BATCH_SIZE = 100_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara token_position standardizzata per Build 6.")
    parser.add_argument("--training-parquet", type=Path, required=True)
    parser.add_argument("--state-arrays-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.training_parquet.is_file():
        raise FileNotFoundError(f"Parquet non trovato: {args.training_parquet}")
    if args.output_dir.exists():
        raise FileExistsError(f"Output gia' esistente: {args.output_dir}")

    expected = {}
    for split in SPLITS:
        labels_path = args.state_arrays_dir / f"y_{split}.npy"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Label non trovate: {labels_path}")
        expected[split] = int(np.load(labels_path, mmap_mode="r").shape[0])

    parquet = pq.ParquetFile(args.training_parquet)
    required = {"split", "token_position"}
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError("Parquet privo di split o token_position.")

    print("1) Calcolo media e deviazione standard train di token_position...")
    counts = {split: 0 for split in SPLITS}
    train_sum = train_square_sum = 0.0
    for batch in parquet.iter_batches(columns=["split", "token_position"], batch_size=BATCH_SIZE):
        splits = np.asarray(batch.column("split").to_pylist(), dtype=object)
        positions = np.asarray(batch.column("token_position").to_numpy(zero_copy_only=False), dtype=np.float64)
        for split in SPLITS:
            mask = splits == split
            counts[split] += int(mask.sum())
            if split == "train" and mask.any():
                values = positions[mask]
                train_sum += float(values.sum())
                train_square_sum += float(np.square(values).sum())
    if counts != expected:
        raise ValueError(f"Conteggi Parquet e label non allineati: {counts} != {expected}")
    mean = train_sum / counts["train"]
    std = float(np.sqrt(max(train_square_sum / counts["train"] - mean**2, 0.0)))
    if not np.isfinite(std) or std == 0.0:
        raise ValueError("Deviazione standard train di token_position non valida.")

    print("2) Materializzo token_position standardizzata in array contigui...")
    arrays_dir = args.output_dir / "arrays"
    arrays_dir.mkdir(parents=True)
    outputs = {
        split: np.lib.format.open_memmap(
            arrays_dir / f"token_position_{split}.npy", mode="w+", dtype=np.float32,
            shape=(counts[split], 1),
        )
        for split in SPLITS
    }
    offsets = {split: 0 for split in SPLITS}
    for batch in parquet.iter_batches(columns=["split", "token_position"], batch_size=BATCH_SIZE):
        splits = np.asarray(batch.column("split").to_pylist(), dtype=object)
        positions = np.asarray(batch.column("token_position").to_numpy(zero_copy_only=False), dtype=np.float64)
        for split in SPLITS:
            mask = splits == split
            amount = int(mask.sum())
            if amount:
                start, stop = offsets[split], offsets[split] + amount
                outputs[split][start:stop, 0] = ((positions[mask] - mean) / std).astype(np.float32)
                offsets[split] = stop
    if offsets != counts:
        raise RuntimeError("Scrittura token_position incompleta.")
    for values in outputs.values():
        values.flush()

    artifacts = args.output_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "preprocessing_summary.json").write_text(json.dumps({
        "experiment": "build6_token_position_inputs",
        "training_parquet": str(args.training_parquet),
        "token_embedding_input": False,
        "split_examples": counts,
        "token_position": {"standardization": "(x - train_mean) / train_std", "train_mean": mean, "train_std": std, "fit_split": "train"},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Token position Build 6 preparata: {args.output_dir}")


if __name__ == "__main__":
    main()
