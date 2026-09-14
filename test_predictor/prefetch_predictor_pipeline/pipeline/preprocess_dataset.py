#!/usr/bin/env python3
"""Create a Parquet table and NumPy arrays from a derived prefetch JSONL file."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


NUM_EXPERTS = 256
TOP_K = 6
ROUTER_SCORES = 16
KV_CURRENT_STAT_NAMES = ("l2_norm", "rms", "mean", "std", "max_abs")
KV_CURRENT_COLUMNS = [f"kv_current_{name}" for name in KV_CURRENT_STAT_NAMES]


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield derived JSONL records without retaining the whole dataset in memory."""
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def load_splits(path: Path) -> dict[int, str]:
    """
    Load a fixed prompt-level split mapping.

    Args:
        path: JSON file containing ``split_per_prompt``.

    Returns:
        Mapping from prompt ID to train, validation or test.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in data["split_per_prompt"].items()}


def kv_current_values(record: dict[str, Any], expected_size: int) -> list[float]:
    """Read the optional direct current-KV statistics in their fixed order."""
    feature = record.get("kv_features_current")
    if expected_size == 0:
        if feature is not None:
            raise ValueError("KV current features are present only in part of the dataset.")
        return []
    if not isinstance(feature, dict) or feature.get("input_dimension") != 512:
        raise ValueError("Expected kv_features_current from a 512D KV row.")
    stats = feature.get("stats_current")
    if not isinstance(stats, dict) or set(stats) != set(KV_CURRENT_STAT_NAMES):
        raise ValueError("Invalid kv_features_current stats_current schema.")
    values = [float(stats[name]) for name in KV_CURRENT_STAT_NAMES]
    if not np.isfinite(values).all():
        raise ValueError("KV current statistics must be finite.")
    return values


def make_row_and_features(
    record: dict[str, Any], split: str, hidden_size: int, kv_current_size: int
) -> tuple[dict[str, Any], list[float]]:
    """Flatten one derived record after its feature widths have been validated."""
    history = record["routing_history"]["expert_ids"][-1]
    logits = record["router_logits"]
    hidden = record.get("hidden_features", {}).get("values", [])
    if len(history) != TOP_K or logits["stored_count"] != ROUTER_SCORES:
        raise ValueError("Expected H=1 Top-6 routing and 16 stored logits.")
    if len(hidden) != hidden_size:
        raise ValueError("All records must have the same hidden feature size.")
    row: dict[str, Any] = {
        "split": split,
        "prompt_id": record["prompt_id"],
        "prompt_category": record["prompt_category"],
        "token_position": record["token_position"],
        "token_id": record["token_id"],
        "layer": record["layer"],
    }
    for index in range(TOP_K):
        row[f"routing_history_expert_{index}"] = history[index]
        row[f"experts_t_{index}"] = record["experts_t"][index]
        row[f"target_expert_{index}"] = record["target_experts_next_token"][index]
    for index in range(ROUTER_SCORES):
        row[f"router_logit_expert_id_{index}"] = logits["expert_ids_ranked"][index]
        row[f"router_logit_value_{index}"] = logits["values_ranked"][index]
    for index, value in enumerate(hidden):
        row[f"hidden_pca_{index}"] = value
    kv_values = kv_current_values(record, kv_current_size)
    for column, value in zip(KV_CURRENT_COLUMNS, kv_values):
        row[column] = value
    return row, [
        record["token_position"], record["token_id"], record["layer"], *history,
        *logits["expert_ids_ranked"], *logits["values_ranked"], *hidden, *kv_values,
    ]


def scan_dataset(path: Path, splits: dict[int, str]) -> tuple[dict[str, int], int, int]:
    """Count examples by split and determine the fixed hidden-feature width."""
    counts = {"train": 0, "validation": 0, "test": 0}
    hidden_size: int | None = None
    kv_current_size: int | None = None
    for record in iter_records(path):
        history = record["routing_history"]["expert_ids"][-1]
        logits = record["router_logits"]
        if len(history) != TOP_K or logits["stored_count"] != ROUTER_SCORES:
            raise ValueError("Expected H=1 Top-6 routing and 16 stored logits.")
        hidden = record.get("hidden_features", {}).get("values", [])
        if hidden_size is None:
            hidden_size = len(hidden)
        if len(hidden) != hidden_size:
            raise ValueError("All records must have the same hidden feature size.")
        feature = record.get("kv_features_current")
        current_size = 0 if feature is None else len(KV_CURRENT_STAT_NAMES)
        if kv_current_size is None:
            kv_current_size = current_size
        if current_size != kv_current_size:
            raise ValueError("KV current features must be present in every record or in none.")
        kv_current_values(record, kv_current_size)
        split = splits[int(record["prompt_id"])]
        if split not in counts:
            raise ValueError(f"Unsupported split: {split}")
        counts[split] += 1
    if hidden_size is None:
        raise ValueError("Input JSONL contains no records.")
    return counts, hidden_size, kv_current_size or 0


def main() -> None:
    """
    Build per-split arrays and one Parquet table from a derived dataset.

    Args:
        No Python arguments. Input, split map and output are command-line options.

    Returns:
        ``None``. Writes Parquet, X/y arrays and a feature-column manifest.
    """
    parser = argparse.ArgumentParser(description="Preprocess a derived prefetch dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-by-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    splits = load_splits(args.split_by_prompt)
    counts, hidden_size, kv_current_size = scan_dataset(args.input, splits)
    arrays_dir = args.output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    feature_size = 3 + TOP_K + 2 * ROUTER_SCORES + hidden_size + kv_current_size
    x_arrays = {
        split: np.lib.format.open_memmap(arrays_dir / f"X_{split}.npy", mode="w+", dtype=np.float32,
                                        shape=(count, feature_size))
        for split, count in counts.items()
    }
    y_arrays = {
        split: np.lib.format.open_memmap(arrays_dir / f"y_{split}.npy", mode="w+", dtype=np.uint8,
                                        shape=(count, NUM_EXPERTS))
        for split, count in counts.items()
    }
    positions = {split: 0 for split in counts}
    parquet_path = args.output_dir / "prefetch_training.parquet"
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []
    for record in iter_records(args.input):
        split = splits[int(record["prompt_id"])]
        row, features = make_row_and_features(record, split, hidden_size, kv_current_size)
        index = positions[split]
        x_arrays[split][index] = features
        y_arrays[split][index] = 0
        y_arrays[split][index, record["target_experts_next_token"]] = 1
        positions[split] += 1
        rows.append(row)
        if len(rows) == 10_000:
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema)
            writer.write_table(table)
            rows.clear()
    if rows:
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()
    for split in ("train", "validation", "test"):
        x_arrays[split].flush()
        y_arrays[split].flush()
    (args.output_dir / "feature_columns.json").write_text(
        json.dumps({"columns": [
            *[f"routing_history_expert_{index}" for index in range(TOP_K)],
            *[f"router_logit_expert_id_{index}" for index in range(ROUTER_SCORES)],
            *[f"router_logit_value_{index}" for index in range(ROUTER_SCORES)],
            *[f"hidden_pca_{index}" for index in range(hidden_size)],
            *(KV_CURRENT_COLUMNS if kv_current_size else []),
        ]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Preprocessing complete: {sum(counts.values())} examples -> {args.output_dir}")


if __name__ == "__main__":
    main()
