#!/usr/bin/env python3
"""Add direct current-KV statistics to a derived prefetch JSONL dataset.

The script streams the input JSONL and reads only one 512D ``kv`` row at a
time from the collector binary files.  It deliberately does not fit PCA,
normalization or any learned transformation: all five output values are
properties of the current KV row itself and are therefore causal for the
prediction of the next token's routing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


KV_DIMENSION = 512
FEATURE_NAMES = ("l2_norm", "rms", "mean", "std", "max_abs")
MAX_OPEN_KV_FILES = 32


def validate_reference(reference: object, line_number: int) -> dict[str, Any]:
    """Validate one collector reference to a current 512D KV row."""
    if not isinstance(reference, dict):
        raise ValueError(f"Line {line_number}: missing kv_raw_ref.")
    if reference.get("dtype") != "float32" or reference.get("shape") != [KV_DIMENSION]:
        raise ValueError(
            f"Line {line_number}: expected kv_raw_ref with dtype float32 and shape [{KV_DIMENSION}]."
        )
    if reference.get("source_tensor") != "kv":
        raise ValueError(f"Line {line_number}: kv_raw_ref source_tensor must be 'kv'.")
    if reference.get("source_stage") != "after_rmsnorm_rope":
        raise ValueError(f"Line {line_number}: unexpected KV source_stage.")
    if not isinstance(reference.get("file"), str) or not reference["file"]:
        raise ValueError(f"Line {line_number}: kv_raw_ref file is missing.")
    if not isinstance(reference.get("offset_bytes"), int) or reference["offset_bytes"] < 0:
        raise ValueError(f"Line {line_number}: invalid kv_raw_ref offset_bytes.")
    return reference


class KVReader:
    """Read fixed-size float32 KV rows while keeping a bounded set of open files."""

    def __init__(self) -> None:
        self._files: OrderedDict[str, BinaryIO] = OrderedDict()

    def read(self, reference: dict[str, Any]) -> np.ndarray:
        path = str(reference["file"])
        handle = self._files.pop(path, None)
        if handle is None:
            handle = Path(path).open("rb")
        self._files[path] = handle
        if len(self._files) > MAX_OPEN_KV_FILES:
            _, oldest = self._files.popitem(last=False)
            oldest.close()
        handle.seek(int(reference["offset_bytes"]))
        vector = np.fromfile(handle, dtype=np.float32, count=KV_DIMENSION)
        if vector.shape != (KV_DIMENSION,):
            raise ValueError(f"Incomplete KV vector at {path}:{reference['offset_bytes']}.")
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite KV value at {path}:{reference['offset_bytes']}.")
        return vector

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()


def current_stats(vector: np.ndarray) -> dict[str, float]:
    """Return direct, population statistics of one current KV vector."""
    values = vector.astype(np.float64, copy=False)
    squared_sum = float(np.dot(values, values))
    return {
        "l2_norm": math.sqrt(squared_sum),
        "rms": math.sqrt(squared_sum / KV_DIMENSION),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "max_abs": float(np.abs(values).max()),
    }


def update_ranges(ranges: dict[str, list[float]], stats: dict[str, float]) -> None:
    """Update min/max diagnostics for the final summary."""
    for name, value in stats.items():
        bounds = ranges[name]
        bounds[0] = min(bounds[0], value)
        bounds[1] = max(bounds[1], value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add direct current-KV statistics to a derived JSONL dataset without PCA."
    )
    parser.add_argument("--input", type=Path, required=True, help="Derived JSONL containing kv_raw_ref.")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL with kv_features_current.")
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON summary path (default: <output stem>_summary.json).",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input JSONL non trovato: {args.input}")
    if args.output.exists():
        raise FileExistsError(f"Output gia' esistente: {args.output}")
    report_path = args.report or args.output.with_name(f"{args.output.stem}_summary.json")
    if report_path.exists():
        raise FileExistsError(f"Report gia' esistente: {report_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    if temporary_output.exists():
        raise FileExistsError(f"Output temporaneo gia' esistente: {temporary_output}")

    reader = KVReader()
    ranges = {name: [math.inf, -math.inf] for name in FEATURE_NAMES}
    rows = 0
    try:
        print("1) Leggo le righe KV raw e calcolo le statistiche current in streaming...")
        with args.input.open(encoding="utf-8") as source, temporary_output.open("w", encoding="utf-8") as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                reference = validate_reference(record.get("kv_raw_ref"), line_number)
                stats = current_stats(reader.read(reference))
                update_ranges(ranges, stats)
                record["kv_features_current"] = {
                    "source_tensor": reference["source_tensor"],
                    "source_stage": reference["source_stage"],
                    "input_dimension": KV_DIMENSION,
                    "stats_current": stats,
                }
                json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
                destination.write("\n")
                rows += 1
        if rows == 0:
            raise ValueError("Input JSONL contains no records.")
        os.replace(temporary_output, args.output)
        report_path.write_text(
            json.dumps(
                {
                    "input": str(args.input),
                    "output": str(args.output),
                    "record_count": rows,
                    "source_tensor": "kv",
                    "source_stage": "after_rmsnorm_rope",
                    "input_dimension": KV_DIMENSION,
                    "feature_names": list(FEATURE_NAMES),
                    "normalization_or_pca_fitted": False,
                    "feature_ranges": {
                        name: {"min": bounds[0], "max": bounds[1]}
                        for name, bounds in ranges.items()
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("2) Scrivo il dataset derivato con kv_features_current...")
        print(f"KV current dataset complete: {rows} examples -> {args.output}")
        print(f"Report: {report_path}")
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    finally:
        reader.close()


if __name__ == "__main__":
    main()
