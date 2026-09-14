#!/usr/bin/env python3
"""Materialize fixed-length state feature slots from existing CSR matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


SPLITS = ("train", "validation", "test")


def source_path(state_arrays_dir: Path, split: str) -> Path:
    """Return the standard state CSR path, including the legacy fallback."""
    path = state_arrays_dir / f"X_{split}.npz"
    if not path.is_file():
        path = state_arrays_dir / f"X_classic_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{split}: matrice state CSR non trovata in {state_arrays_dir}")
    return path


def write_split(
    matrix: sparse.csr_matrix,
    split: str,
    arrays_dir: Path,
    chunk_rows: int,
    expected_nonzero_features: int | None,
) -> int:
    """Write fixed feature slots plus one final bias sentinel for one split."""
    matrix = matrix.tocsr().astype(np.float32)
    nonzero_counts = np.diff(matrix.indptr)
    if nonzero_counts.size == 0 or np.any(nonzero_counts != nonzero_counts[0]):
        raise ValueError(f"{split}: ogni esempio deve avere lo stesso numero di feature non nulle.")
    nonzero_features = int(nonzero_counts[0])
    if expected_nonzero_features is not None and nonzero_features != expected_nonzero_features:
        raise ValueError(
            f"{split}: {nonzero_features} feature non nulle; attese {expected_nonzero_features}."
        )
    feature_count = matrix.shape[1]
    if feature_count > np.iinfo(np.uint16).max:
        raise ValueError("Sono supportate al massimo 65.535 feature state.")
    if matrix.indices.min(initial=0) < 0 or matrix.indices.max(initial=0) >= feature_count:
        raise ValueError(f"{split}: indice CSR fuori intervallo.")

    rows = matrix.shape[0]
    slots = nonzero_features + 1
    indices_path = arrays_dir / f"state_feature_indices_{split}.npy"
    values_path = arrays_dir / f"state_feature_values_{split}.npy"
    output_indices = np.lib.format.open_memmap(
        indices_path, mode="w+", dtype=np.uint16, shape=(rows, slots)
    )
    output_values = np.lib.format.open_memmap(
        values_path, mode="w+", dtype=np.float32, shape=(rows, slots)
    )
    flat_indices = matrix.indices.reshape(rows, nonzero_features)
    flat_values = matrix.data.reshape(rows, nonzero_features)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        output_indices[start:stop, :-1] = flat_indices[start:stop]
        output_values[start:stop, :-1] = flat_values[start:stop]
        output_indices[start:stop, -1] = feature_count
        output_values[start:stop, -1] = 1.0
    output_indices.flush()
    output_values.flush()
    return nonzero_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materializza gli slot state fixed-NNZ dagli array CSR esistenti."
    )
    parser.add_argument("--state-arrays-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows deve essere positivo.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output gia' esistente: {args.output_dir}")

    arrays_dir = args.output_dir / "arrays"
    artifacts_dir = args.output_dir / "artifacts"
    arrays_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    expected_nonzero_features: int | None = None
    summary: dict[str, object] = {"splits": {}}
    feature_count: int | None = None

    print("1) Converto le matrici CSR esistenti in slot state contigui fixed-NNZ...")
    for split in SPLITS:
        path = source_path(args.state_arrays_dir, split)
        matrix = sparse.load_npz(path).tocsr()
        if feature_count is None:
            feature_count = matrix.shape[1]
        elif matrix.shape[1] != feature_count:
            raise ValueError(f"{split}: numero di feature state diverso dagli altri split.")
        print(f"   {split}: {matrix.shape[0]:,} esempi da {path.name}...")
        expected_nonzero_features = write_split(
            matrix, split, arrays_dir, args.chunk_rows, expected_nonzero_features
        )
        summary["splits"][split] = {
            "examples": matrix.shape[0],
            "source_nnz": int(matrix.nnz),
        }

    assert feature_count is not None and expected_nonzero_features is not None
    metadata = {
        "format": "fixed_nnz_state_slots_v1",
        "feature_count": feature_count,
        "nonzero_features_per_example": expected_nonzero_features,
        "input_slots": expected_nonzero_features + 1,
        "feature_indices_dtype": "uint16",
        "feature_values_dtype": "float32",
        "bias_slot": {
            "index": feature_count,
            "value": 1.0,
            "purpose": "Rappresenta il bias del Dense state originale.",
        },
        "source_state_arrays_dir": str(args.state_arrays_dir),
    }
    (artifacts_dir / "fixed_nnz_state_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    summary.update(metadata)
    (args.output_dir / "fixed_nnz_state_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\nState fixed-NNZ materializzato.")
    print(f"Output: {args.output_dir}")
    print(
        f"   feature={feature_count}; feature non nulle/esempio={expected_nonzero_features}; "
        f"slot input={expected_nonzero_features + 1} (incluso bias)."
    )


if __name__ == "__main__":
    main()
