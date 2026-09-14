#!/usr/bin/env python3
"""Quantize normalized float16 hidden arrays to calibrated symmetric int8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HIDDEN_DIMENSION = 4096
SPLITS = ("train", "validation", "test")
DEFAULT_BATCH_SIZE = 4096
DEFAULT_SAMPLE_ROWS = 4096
DEFAULT_PERCENTILE = 99.99


def load_hidden(input_dir: Path, split: str) -> np.ndarray:
    path = input_dir / "arrays" / f"raw_hidden_normalized_{split}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Hidden float16 mancanti: {path}")
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2 or values.shape[1] != HIDDEN_DIMENSION or values.dtype != np.float16:
        raise ValueError(f"{split}: atteso array float16 [N, {HIDDEN_DIMENSION}].")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantizza hidden normalizzati float16 in int8.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--calibration-sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--calibration-percentile", type=float, default=DEFAULT_PERCENTILE)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"La directory di output esiste gia': {args.output_dir}")
    if args.batch_size < 1 or args.calibration_sample_rows < 1 or not 90.0 <= args.calibration_percentile < 100.0:
        parser.error("Batch/sample devono essere positivi e percentile in [90, 100).")

    train = load_hidden(args.input_dir, "train")
    sample_count = min(args.calibration_sample_rows, len(train))
    sample_rows = np.linspace(0, len(train) - 1, sample_count, dtype=np.int64)
    print(f"1) Calibro la scala sul solo train con {sample_count:,} righe...")
    sample = np.asarray(train[sample_rows], dtype=np.float32)
    clipping_limit = float(np.percentile(np.abs(sample), args.calibration_percentile))
    del sample
    if not np.isfinite(clipping_limit) or clipping_limit <= 0:
        raise ValueError("Limite di clipping non valido.")
    scale = clipping_limit / 127.0
    print(f"   percentile {args.calibration_percentile}: limite ±{clipping_limit:.6g}, scala={scale:.6g}")

    arrays_dir = args.output_dir / "arrays"
    artifacts_dir = args.output_dir / "artifacts"
    arrays_dir.mkdir(parents=True)
    artifacts_dir.mkdir()
    split_reports: dict[str, object] = {}
    for split in SPLITS:
        source = train if split == "train" else load_hidden(args.input_dir, split)
        destination_path = arrays_dir / f"raw_hidden_normalized_int8_{split}.npy"
        destination = np.lib.format.open_memmap(
            destination_path, mode="w+", dtype=np.int8, shape=source.shape
        )
        saturated = 0
        squared_error = 0.0
        value_count = int(source.shape[0] * source.shape[1])
        print(f"2) Quantizzo {split}: {source.shape[0]:,} vettori...")
        for start in range(0, len(source), args.batch_size):
            stop = min(start + args.batch_size, len(source))
            values = np.asarray(source[start:stop], dtype=np.float32)
            saturated += int(np.count_nonzero(np.abs(values) >= clipping_limit))
            quantized = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
            destination[start:stop] = quantized
            reconstructed = quantized.astype(np.float32) * scale
            squared_error += float(np.square(values - reconstructed).sum(dtype=np.float64))
        destination.flush()
        split_reports[split] = {
            "examples": int(source.shape[0]),
            "path": str(destination_path),
            "saturated_values": saturated,
            "saturation_fraction": saturated / value_count,
            "reconstruction_rmse": float(np.sqrt(squared_error / value_count)),
        }
    summary = {
        "experiment": "symmetric_int8_normalized_raw_hidden",
        "source_float16_dir": str(args.input_dir),
        "calibration_split": "train",
        "calibration_sample_rows": sample_count,
        "calibration_percentile": args.calibration_percentile,
        "clipping_limit": clipping_limit,
        "dequantization_scale": scale,
        "formula": "q=clip(round(h/scale),-127,127); h_approx=q*scale",
        "splits": split_reports,
    }
    (artifacts_dir / "quantization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nHidden int8 completati. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
