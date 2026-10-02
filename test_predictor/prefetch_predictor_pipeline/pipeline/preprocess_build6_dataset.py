#!/usr/bin/env python3
"""Create Build 6 Parquet rows and aligned multi-label arrays.

Build 6 keeps current state columns and adds the matching state of layer l-1.
Raw hidden and full KV remain references in the derived JSONL: specialised
materializers create their contiguous training arrays later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SPLITS = ("train", "validation", "test")
TOP_K = 6
ROUTER_SCORES = 16
NUM_EXPERTS = 256
KV_STAT_NAMES = ("l2_norm", "rms", "mean", "std", "max_abs")


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def load_splits(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = {int(key): str(value) for key, value in payload["split_per_prompt"].items()}
    if not values or any(value not in SPLITS for value in values.values()):
        raise ValueError("split_by_prompt non valido.")
    return values


def state_values(record: dict[str, Any], context: str) -> tuple[list[int], list[int], list[float], list[float]]:
    """Return H=1 history, router IDs/values and five KV statistics."""
    try:
        history = record["routing_history"]["expert_ids"][-1]
        logits = record["router_logits"]
        router_ids = logits["expert_ids_ranked"]
        router_values = logits["values_ranked"]
        stats = record["kv_features_current"]["stats_current"]
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"State {context} incompleto.") from error
    values = [float(stats[name]) for name in KV_STAT_NAMES]
    if (len(history) != TOP_K or len(router_ids) != ROUTER_SCORES or len(router_values) != ROUTER_SCORES
            or any(not 0 <= int(value) < NUM_EXPERTS for value in [*history, *router_ids])
            or not np.isfinite([*router_values, *values]).all()):
        raise ValueError(f"State {context} non valido.")
    return [int(value) for value in history], [int(value) for value in router_ids], [float(value) for value in router_values], values


def make_row(record: dict[str, Any], split: str) -> dict[str, Any]:
    previous = record.get("previous_layer")
    if not isinstance(previous, dict) or int(previous.get("layer", -1)) != int(record["layer"]) - 1:
        raise ValueError("previous_layer assente o non allineato a layer-1.")
    current_history, current_ids, current_values, current_kv = state_values(record, "corrente")
    previous_history, previous_ids, previous_values, previous_kv = state_values(previous, "precedente")
    target = record.get("target_experts_next_token")
    if not isinstance(target, list) or len(target) != TOP_K or len(set(target)) != TOP_K:
        raise ValueError("Target esperti non valido.")
    row: dict[str, Any] = {
        "split": split,
        "prompt_id": int(record["prompt_id"]),
        "prompt_category": str(record["prompt_category"]),
        "token_position": float(record["token_position"]),
        "token_id": int(record["token_id"]),
        "layer": int(record["layer"]),
        "previous_layer": int(previous["layer"]),
    }
    for prefix, history, ids, values, kv in (
        ("", current_history, current_ids, current_values, current_kv),
        ("previous_", previous_history, previous_ids, previous_values, previous_kv),
    ):
        for index, value in enumerate(history):
            row[f"{prefix}routing_history_expert_{index}"] = value
        for index, value in enumerate(ids):
            row[f"{prefix}router_logit_expert_id_{index}"] = value
        for index, value in enumerate(values):
            row[f"{prefix}router_logit_value_{index}"] = value
        for name, value in zip(KV_STAT_NAMES, kv):
            row[f"{prefix}kv_current_{name}"] = value
    for index, value in enumerate(target):
        row[f"target_expert_{index}"] = int(value)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessa il derived dataset Build 6 in Parquet e label.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-by-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file() or not args.split_by_prompt.is_file():
        raise FileNotFoundError("Derived input o split_by_prompt non trovato.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output gia' esistente: {args.output_dir}")

    assignments = load_splits(args.split_by_prompt)
    counts = {split: 0 for split in SPLITS}
    print("1) Conteggio degli esempi Build 6 per split...")
    for record in iter_records(args.input):
        split = assignments.get(int(record["prompt_id"]))
        if split is None:
            raise ValueError("Prompt assente dallo split.")
        counts[split] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Uno split Build 6 e' vuoto: {counts}")

    args.output_dir.mkdir(parents=True)
    arrays_dir = args.output_dir / "arrays"
    arrays_dir.mkdir()
    labels = {
        split: np.lib.format.open_memmap(arrays_dir / f"y_{split}.npy", mode="w+", dtype=np.uint8, shape=(count, NUM_EXPERTS))
        for split, count in counts.items()
    }
    positions = {split: 0 for split in SPLITS}
    parquet_path = args.output_dir / "prefetch_training.parquet"
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []
    print("2) Scrittura Parquet e label Build 6 allineate...")
    for record in iter_records(args.input):
        split = assignments[int(record["prompt_id"])]
        row = make_row(record, split)
        position = positions[split]
        labels[split][position] = 0
        labels[split][position, [row[f"target_expert_{index}"] for index in range(TOP_K)]] = 1
        positions[split] += 1
        rows.append(row)
        if len(rows) >= 10_000:
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
    for split in SPLITS:
        if positions[split] != counts[split] or not np.all(labels[split].sum(axis=1) == TOP_K):
            raise ValueError(f"Label Build 6 non valide per {split}.")
        labels[split].flush()
    (args.output_dir / "preprocessing_summary.json").write_text(
        json.dumps({"experiment": "build_dataset_6_preprocess", "input": str(args.input), "split_by_prompt": str(args.split_by_prompt), "target_layers": [4, 42], "splits": counts, "token_embedding_input": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Preprocessing Build 6 completato: {sum(counts.values()):,} esempi -> {args.output_dir}")


if __name__ == "__main__":
    main()
