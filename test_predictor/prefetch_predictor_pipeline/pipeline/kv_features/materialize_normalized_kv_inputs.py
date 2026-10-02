#!/usr/bin/env python3
"""Materialize train-normalized current 512D KV rows into contiguous float16 arrays.

The raw KV cache rows remain untouched in the derived JSONL.  This script creates
an aligned, efficient training view only: per-layer mean and standard deviation
are fitted on train prompts alone, then applied to train, validation and test.
No PCA or learned KV projection is performed here.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import numpy as np


KV_DIMENSION = 512
NUM_LAYERS = 40
FIRST_LAYER = 3
SPLITS = ("train", "validation", "test")
DEFAULT_BATCH_SIZE = 4096


def load_prompt_splits(path: Path) -> dict[int, str]:
    """Return the prompt-id to split assignment from the project split JSON."""
    content = json.loads(path.read_text(encoding="utf-8"))
    split_per_prompt = content.get("split_per_prompt")
    if not isinstance(split_per_prompt, dict):
        raise ValueError("Split per prompt privo del campo split_per_prompt.")
    assignments = {int(prompt_id): str(split) for prompt_id, split in split_per_prompt.items()}
    if any(split not in SPLITS for split in assignments.values()):
        raise ValueError("Lo split contiene un valore non supportato.")
    if not assignments:
        raise ValueError("Split per prompt vuoto o non valido.")
    return assignments


def expected_counts(state_arrays_dir: Path) -> dict[str, int]:
    """Read expected row counts from the state-label arrays preserving alignment."""
    counts: dict[str, int] = {}
    for split in SPLITS:
        path = state_arrays_dir / f"y_{split}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Label non trovate: {path}")
        labels = np.load(path, mmap_mode="r")
        if labels.ndim != 2 or labels.shape[1] != 256:
            raise ValueError(f"{split}: label state non valide in {path}")
        counts[split] = int(labels.shape[0])
    return counts


def resolve_path(record: dict[str, Any], path: str, line_number: int) -> Any:
    """Read a dotted JSON path, preserving the Build 5 default fields."""
    value: Any = record
    for name in path.split("."):
        if not isinstance(value, dict) or name not in value:
            raise ValueError(f"Line {line_number}: campo {path} mancante.")
        value = value[name]
    return value


def validate_record(record: dict[str, Any], line_number: int, reference_path: str, layer_path: str) -> tuple[int, int, dict[str, Any]]:
    """Validate the fields needed to materialize one current KV row."""
    try:
        prompt_id = int(record["prompt_id"])
        layer_number = int(resolve_path(record, layer_path, line_number))
        reference = resolve_path(record, reference_path, line_number)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Line {line_number}: record KV incompleto.") from error
    if not FIRST_LAYER <= layer_number < FIRST_LAYER + NUM_LAYERS:
        raise ValueError(f"Line {line_number}: layer fuori intervallo: {layer_number}")
    if not isinstance(reference, dict):
        raise ValueError(f"Line {line_number}: kv_raw_ref mancante.")
    if reference.get("dtype") != "float32" or reference.get("shape") != [KV_DIMENSION]:
        raise ValueError(f"Line {line_number}: kv_raw_ref deve essere float32[{KV_DIMENSION}].")
    if reference.get("source_tensor") != "kv" or reference.get("source_stage") != "after_rmsnorm_rope":
        raise ValueError(f"Line {line_number}: riferimento KV con sorgente o stadio inatteso.")
    if not isinstance(reference.get("file"), str) or not reference["file"]:
        raise ValueError(f"Line {line_number}: file KV mancante.")
    if not isinstance(reference.get("offset_bytes"), int) or reference["offset_bytes"] < 0:
        raise ValueError(f"Line {line_number}: offset KV non valido.")
    return prompt_id, layer_number - FIRST_LAYER, reference


class KVReader:
    """Read KV rows through a bounded LRU cache of binary files."""

    def __init__(self, max_open_files: int = 32) -> None:
        self.max_open_files = max_open_files
        self.files: OrderedDict[str, object] = OrderedDict()

    def read_batch(self, references: list[dict[str, Any]]) -> np.ndarray:
        values = np.empty((len(references), KV_DIMENSION), dtype=np.float32)
        for row, reference in enumerate(references):
            path = str(reference["file"])
            handle = self.files.pop(path, None)
            if handle is None:
                handle = Path(path).open("rb")
                if len(self.files) >= self.max_open_files:
                    self.files.popitem(last=False)[1].close()
            self.files[path] = handle
            handle.seek(int(reference["offset_bytes"]))
            vector = np.fromfile(handle, dtype=np.float32, count=KV_DIMENSION)
            if vector.shape != (KV_DIMENSION,) or not np.isfinite(vector).all():
                raise ValueError(f"Vettore KV incompleto o non finito: {path}:{reference['offset_bytes']}")
            values[row] = vector
        return values

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()
        self.files.clear()


def iter_records(input_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield non-empty JSON records with their one-based source line number."""
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def fit_train_statistics(input_path: Path, assignments: dict[int, str], batch_size: int, reference_path: str, layer_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit exact per-component train-only mean/std, batching disk reads."""
    counts = np.zeros(NUM_LAYERS, dtype=np.int64)
    sums = np.zeros((NUM_LAYERS, KV_DIMENSION), dtype=np.float64)
    squared_sums = np.zeros((NUM_LAYERS, KV_DIMENSION), dtype=np.float64)
    pending_layers: list[int] = []
    pending_refs: list[dict[str, Any]] = []
    reader = KVReader()

    def consume() -> None:
        if not pending_refs:
            return
        vectors = reader.read_batch(pending_refs).astype(np.float64, copy=False)
        layers = np.asarray(pending_layers, dtype=np.int16)
        for layer_index in np.unique(layers):
            subset = vectors[layers == layer_index]
            counts[layer_index] += len(subset)
            sums[layer_index] += subset.sum(axis=0)
            squared_sums[layer_index] += np.square(subset).sum(axis=0)
        pending_layers.clear()
        pending_refs.clear()

    try:
        for line_number, record in iter_records(input_path):
            prompt_id, layer_index, reference = validate_record(record, line_number, reference_path, layer_path)
            split = assignments.get(prompt_id)
            if split is None:
                raise ValueError(f"Line {line_number}: prompt {prompt_id} assente dallo split.")
            if split == "train":
                pending_layers.append(layer_index)
                pending_refs.append(reference)
                if len(pending_refs) >= batch_size:
                    consume()
        consume()
    finally:
        reader.close()

    observed = counts > 0
    means = np.zeros_like(sums)
    means[observed] = sums[observed] / counts[observed, None]
    variances = np.zeros_like(sums)
    variances[observed] = np.maximum(
        squared_sums[observed] / counts[observed, None] - np.square(means[observed]), 0.0
    )
    stds = np.sqrt(variances)
    stds = np.maximum(stds, np.float64(1e-6))
    return means.astype(np.float32), stds.astype(np.float32), counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Materializza KV current normalizzati train-only in array float16 contigui.")
    parser.add_argument("--input", type=Path, required=True, help="Derived JSONL con kv_raw_ref.")
    parser.add_argument("--split-by-prompt", type=Path, required=True)
    parser.add_argument("--state-arrays-dir", type=Path, required=True, help="Array state per verificare righe e ordine.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-path", default="kv_raw_ref", help="Campo JSON del riferimento KV; supporta percorsi con punti.")
    parser.add_argument("--layer-path", default="layer", help="Campo JSON del layer associato al riferimento KV.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if not args.input.is_file() or not args.split_by_prompt.is_file():
        raise FileNotFoundError("Input JSONL o split per prompt non trovato.")
    if args.batch_size < 1:
        parser.error("--batch-size deve essere positivo.")
    if args.output_dir.exists():
        raise FileExistsError(f"La directory di output esiste gia': {args.output_dir}")

    assignments = load_prompt_splits(args.split_by_prompt)
    counts = expected_counts(args.state_arrays_dir)
    print("1) Calcolo media e deviazione standard KV sul solo train, per layer...")
    means, stds, train_counts_by_layer = fit_train_statistics(args.input, assignments, args.batch_size, args.reference_path, args.layer_path)

    arrays_dir = args.output_dir / "arrays"
    artifacts_dir = args.output_dir / "artifacts"
    args.output_dir.mkdir(parents=True)
    arrays_dir.mkdir()
    artifacts_dir.mkdir()
    outputs = {
        split: np.lib.format.open_memmap(
            arrays_dir / f"kv_normalized_{split}.npy",
            mode="w+",
            dtype=np.float16,
            shape=(counts[split], KV_DIMENSION),
        )
        for split in SPLITS
    }
    positions = {split: 0 for split in SPLITS}
    pending: list[tuple[str, int, dict[str, Any]]] = []
    reader = KVReader()

    def write_pending() -> None:
        if not pending:
            return
        vectors = reader.read_batch([reference for _, _, reference in pending])
        for vector, (split, layer_index, _) in zip(vectors, pending):
            position = positions[split]
            if position >= counts[split]:
                raise ValueError(f"{split}: piu record KV degli array state.")
            outputs[split][position] = ((vector - means[layer_index]) / stds[layer_index]).astype(np.float16)
            positions[split] += 1
        pending.clear()

    try:
        print("2) Materializzo KV 512D normalizzati e allineati agli array state...")
        for line_number, record in iter_records(args.input):
            prompt_id, layer_index, reference = validate_record(record, line_number, args.reference_path, args.layer_path)
            split = assignments.get(prompt_id)
            if split is None:
                raise ValueError(f"Line {line_number}: prompt {prompt_id} assente dallo split.")
            pending.append((split, layer_index, reference))
            if len(pending) >= args.batch_size:
                write_pending()
        write_pending()
    finally:
        reader.close()

    for split in SPLITS:
        if positions[split] != counts[split]:
            raise ValueError(f"{split}: scritti {positions[split]} KV, attesi {counts[split]}.")
        outputs[split].flush()
    np.save(artifacts_dir / "kv_train_mean_by_layer.npy", means)
    np.save(artifacts_dir / "kv_train_std_by_layer.npy", stds)
    (artifacts_dir / "materialization_summary.json").write_text(
        json.dumps(
            {
                "experiment": "materialized_normalized_current_kv_inputs",
                "source_input": str(args.input),
                "reference_path": args.reference_path,
                "layer_path": args.layer_path,
                "split_by_prompt": str(args.split_by_prompt),
                "state_arrays_dir": str(args.state_arrays_dir),
                "source_tensor": "kv",
                "source_stage": "after_rmsnorm_rope",
                "input_dimension": KV_DIMENSION,
                "storage_dtype": "float16",
                "normalization": "exact per-component mean/std fit only on train rows, separately for each layer; std floor=1e-6",
                "pca_or_learned_projection": False,
                "train_examples_by_layer": train_counts_by_layer.tolist(),
                "splits": {split: {"examples": counts[split], "path": str(arrays_dir / f'kv_normalized_{split}.npy')} for split in SPLITS},
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nKV normalizzati materializzati. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
