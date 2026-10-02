#!/usr/bin/env python3
"""Extract the current-layer block from Build 7 fixed-NNZ state inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "validation", "test")
CURRENT_FEATURE_COUNT = 4411
CURRENT_NONZERO_FEATURES = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materializza lo state corrente l di Build 7 in 45 slot fixed-NNZ."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows deve essere positivo.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output gia' esistente: {args.output_dir}")
    metadata_path = args.input_dir / "artifacts" / "fixed_nnz_state_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_count") != 13157 or metadata.get("input_slots") != 131:
        raise ValueError("Input non compatibile con lo state Build 7 completo atteso.")
    arrays_dir = args.output_dir / "arrays"
    artifacts_dir = args.output_dir / "artifacts"
    arrays_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    print("1) Estraggo il solo blocco state corrente l dai 131 slot Build 7...")
    split_summary: dict[str, object] = {}
    for split in SPLITS:
        input_indices = np.load(
            args.input_dir / "arrays" / f"state_feature_indices_{split}.npy", mmap_mode="r"
        )
        input_values = np.load(
            args.input_dir / "arrays" / f"state_feature_values_{split}.npy", mmap_mode="r"
        )
        rows = input_indices.shape[0]
        if input_indices.shape != (rows, 131) or input_values.shape != (rows, 131):
            raise ValueError(f"{split}: shape fixed-NNZ Build 7 completa non valida.")
        output_indices = np.lib.format.open_memmap(
            arrays_dir / f"state_feature_indices_{split}.npy", mode="w+",
            dtype=np.uint16, shape=(rows, CURRENT_NONZERO_FEATURES + 1),
        )
        output_values = np.lib.format.open_memmap(
            arrays_dir / f"state_feature_values_{split}.npy", mode="w+",
            dtype=np.float32, shape=(rows, CURRENT_NONZERO_FEATURES + 1),
        )
        print(f"   {split}: {rows:,} righe...")
        for start in range(0, rows, args.chunk_rows):
            stop = min(start + args.chunk_rows, rows)
            indices = np.asarray(input_indices[start:stop])
            values = np.asarray(input_values[start:stop])
            current_mask = indices < CURRENT_FEATURE_COUNT
            if not np.all(current_mask.sum(axis=1) == CURRENT_NONZERO_FEATURES):
                raise ValueError(f"{split}: numero di feature state correnti inatteso.")
            output_indices[start:stop, :-1] = indices[current_mask].reshape(-1, CURRENT_NONZERO_FEATURES)
            output_values[start:stop, :-1] = values[current_mask].reshape(-1, CURRENT_NONZERO_FEATURES)
            output_indices[start:stop, -1] = CURRENT_FEATURE_COUNT
            output_values[start:stop, -1] = 1.0
        output_indices.flush()
        output_values.flush()
        split_summary[split] = {"examples": rows, "input_slots": 45}
    output_metadata = {
        "format": "fixed_nnz_state_slots_v1",
        "feature_count": CURRENT_FEATURE_COUNT,
        "nonzero_features_per_example": CURRENT_NONZERO_FEATURES,
        "input_slots": CURRENT_NONZERO_FEATURES + 1,
        "feature_indices_dtype": "uint16",
        "feature_values_dtype": "float32",
        "bias_slot": {"index": CURRENT_FEATURE_COUNT, "value": 1.0},
        "source": "Build 7 current block l extracted from 02_current_previous_previous_fixed_nnz_state",
    }
    (artifacts_dir / "fixed_nnz_state_metadata.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "current_only_fixed_nnz_state_summary.json").write_text(
        json.dumps({"splits": split_summary, **output_metadata}, indent=2) + "\n", encoding="utf-8"
    )
    print("State corrente Build 7 materializzato.")
    print(f"Output: {args.output_dir}")
    print("   feature=4411; feature non nulle/esempio=44; slot input=45 (incluso bias).")


if __name__ == "__main__":
    main()
