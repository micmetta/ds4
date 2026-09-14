#!/usr/bin/env python3
"""Materialize train-normalized raw hidden vectors into contiguous float16 arrays."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


HIDDEN_DIMENSION = 4096
SPLITS = ("train", "validation", "test")
DEFAULT_BATCH_SIZE = 4096


class HiddenFileReader:
    """Read raw collector vectors through a bounded cache of open files."""

    def __init__(self, paths: list[str], max_open_files: int = 64) -> None:
        self.paths = paths
        self.max_open_files = max_open_files
        self.files: OrderedDict[int, object] = OrderedDict()

    def read_batch(self, file_indices: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        values = np.empty((len(file_indices), HIDDEN_DIMENSION), dtype=np.float32)
        for row, (file_index, offset) in enumerate(zip(file_indices, offsets)):
            index = int(file_index)
            handle = self.files.pop(index, None)
            if handle is None:
                handle = Path(self.paths[index]).open("rb")
                if len(self.files) >= self.max_open_files:
                    self.files.popitem(last=False)[1].close()
            self.files[index] = handle
            handle.seek(int(offset))
            vector = np.fromfile(handle, dtype=np.float32, count=HIDDEN_DIMENSION)
            if vector.shape != (HIDDEN_DIMENSION,):
                raise ValueError(f"Vettore hidden incompleto: {self.paths[index]}:{offset}")
            values[row] = vector
        return values

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()
        self.files.clear()


def load_inputs(input_dir: Path, split: str):
    arrays_dir = input_dir / "arrays"
    artifacts_dir = input_dir / "artifacts"
    file_indices = np.load(arrays_dir / f"hidden_file_index_{split}.npy", mmap_mode="r")
    offsets = np.load(arrays_dir / f"hidden_offset_bytes_{split}.npy", mmap_mode="r")
    layers = np.load(arrays_dir / f"hidden_layer_index_{split}.npy", mmap_mode="r")
    paths = json.loads((artifacts_dir / "hidden_file_paths.json").read_text(encoding="utf-8"))
    means = np.load(artifacts_dir / "hidden_train_mean_by_layer.npy", mmap_mode="r")
    stds = np.load(artifacts_dir / "hidden_train_std_by_layer.npy", mmap_mode="r")
    if any(values.ndim != 1 for values in (file_indices, offsets, layers)):
        raise ValueError(f"{split}: riferimenti raw hidden non vettoriali.")
    if not (len(file_indices) == len(offsets) == len(layers)) or means.shape != (40, HIDDEN_DIMENSION) or stds.shape != (40, HIDDEN_DIMENSION):
        raise ValueError(f"{split}: input raw hidden non coerenti.")
    return file_indices, offsets, layers, paths, means, stds


def main() -> None:
    parser = argparse.ArgumentParser(description="Materializza hidden normalizzati train-only in array float16 contigui.")
    parser.add_argument("--raw-hidden-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size deve essere positivo.")
    if args.output_dir.exists():
        raise FileExistsError(f"La directory di output esiste gia': {args.output_dir}")

    print("1) Verifico i riferimenti e creo gli array hidden contigui float16...")
    args.output_dir.mkdir(parents=True)
    arrays_dir = args.output_dir / "arrays"
    artifacts_dir = args.output_dir / "artifacts"
    arrays_dir.mkdir()
    artifacts_dir.mkdir()
    summary: dict[str, object] = {
        "experiment": "materialized_normalized_raw_hidden_inputs",
        "source_raw_hidden_input_dir": str(args.raw_hidden_input_dir),
        "storage_dtype": "float16",
        "shape_per_example": [HIDDEN_DIMENSION],
        "normalization": "already fit train-only per layer by source 06",
        "splits": {},
    }
    for split in SPLITS:
        file_indices, offsets, layers, paths, means, stds = load_inputs(args.raw_hidden_input_dir, split)
        output_path = arrays_dir / f"raw_hidden_normalized_{split}.npy"
        output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=(len(file_indices), HIDDEN_DIMENSION))
        # Read collector values in physical file/offset order once. The result is
        # written back at the original row, preserving alignment with state arrays.
        order = np.lexsort((offsets, file_indices))
        reader = HiddenFileReader(paths)
        print(f"2) Materializzo {split}: {len(file_indices):,} vettori in batch da {args.batch_size}...")
        try:
            for start in range(0, len(order), args.batch_size):
                rows = order[start : start + args.batch_size]
                batch_layers = layers[rows]
                raw = reader.read_batch(file_indices[rows], offsets[rows])
                normalized = (raw - means[batch_layers]) / stds[batch_layers]
                output[rows] = normalized.astype(np.float16)
        finally:
            reader.close()
        output.flush()
        summary["splits"][split] = {"examples": int(len(file_indices)), "path": str(output_path)}
    (artifacts_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nHidden normalizzati materializzati. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
