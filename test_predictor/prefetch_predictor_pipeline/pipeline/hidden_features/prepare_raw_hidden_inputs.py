#!/usr/bin/env python3
"""Prepare bounded raw-hidden references for the supervised hidden projection FFN."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


HIDDEN_DIMENSION = 4096
ROUTED_LAYERS = tuple(range(3, 43))
SPLITS = ("train", "validation", "test")
DEFAULT_BATCH_SIZE = 4096


class HiddenFileReader:
    """Read raw vectors while limiting the number of open collector files."""

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


def load_split_mapping(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(prompt_id): split for prompt_id, split in payload["split_per_prompt"].items()}


def expected_counts(state_arrays_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        path = state_arrays_dir / f"y_{split}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Label non trovate: {path}")
        counts[split] = int(np.load(path, mmap_mode="r").shape[0])
    return counts


def resolve_path(record: dict, path: str, line_number: int) -> object:
    """Read a dotted JSON path, e.g. ``previous_layer.hidden_raw_ref``."""
    value: object = record
    for name in path.split("."):
        if not isinstance(value, dict) or name not in value:
            raise ValueError(f"Linea {line_number}: campo {path} mancante.")
        value = value[name]
    return value


def validate_reference(record: dict, reference_path: str, line_number: int) -> tuple[str, int]:
    reference = resolve_path(record, reference_path, line_number)
    if not isinstance(reference, dict):
        raise ValueError(f"Linea {line_number}: hidden_raw_ref mancante.")
    if reference.get("dtype") != "float32" or reference.get("shape") != [HIDDEN_DIMENSION]:
        raise ValueError(f"Linea {line_number}: atteso hidden_raw_ref float32[{HIDDEN_DIMENSION}].")
    return str(reference["file"]), int(reference["offset_bytes"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara riferimenti raw hidden e normalizzazione train-only per la FFN."
    )
    parser.add_argument("--raw-hidden-input", type=Path, required=True)
    parser.add_argument("--split-by-prompt", type=Path, required=True)
    parser.add_argument("--state-arrays-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-path", default="hidden_raw_ref", help="Campo JSON del riferimento hidden; supporta percorsi con punti.")
    parser.add_argument("--layer-path", default="layer", help="Campo JSON del layer associato al riferimento hidden.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size deve essere positivo.")
    if args.output_dir.exists():
        raise FileExistsError(f"La directory di output esiste gia': {args.output_dir}")
    if not args.raw_hidden_input.is_file() or not args.split_by_prompt.is_file():
        raise FileNotFoundError("Input raw hidden o split_by_prompt non trovato.")

    split_mapping = load_split_mapping(args.split_by_prompt)
    counts = expected_counts(args.state_arrays_dir)
    records: dict[str, list[tuple[str, int, int]]] = {split: [] for split in SPLITS}
    print("1) Indicizzo i riferimenti raw hidden nell'ordine degli array state...")
    with args.raw_hidden_input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            split = split_mapping.get(int(record["prompt_id"]))
            layer = int(resolve_path(record, args.layer_path, line_number))
            if split not in records or layer not in ROUTED_LAYERS:
                raise ValueError(f"Linea {line_number}: split o layer non valido.")
            path, offset = validate_reference(record, args.reference_path, line_number)
            records[split].append((path, offset, layer - ROUTED_LAYERS[0]))
    observed = {split: len(values) for split, values in records.items()}
    if observed != counts:
        raise ValueError(f"Conteggi raw hidden incompatibili con state arrays: {observed} != {counts}")

    file_paths = sorted({path for values in records.values() for path, _, _ in values})
    file_index = {path: index for index, path in enumerate(file_paths)}
    arrays_dir = args.output_dir / "arrays"
    artifacts_dir = args.output_dir / "artifacts"
    arrays_dir.mkdir(parents=True)
    artifacts_dir.mkdir()
    for split, values in records.items():
        np.save(arrays_dir / f"hidden_file_index_{split}.npy", np.asarray([file_index[path] for path, _, _ in values], dtype=np.int32))
        np.save(arrays_dir / f"hidden_offset_bytes_{split}.npy", np.asarray([offset for _, offset, _ in values], dtype=np.int64))
        np.save(arrays_dir / f"hidden_layer_index_{split}.npy", np.asarray([layer for _, _, layer in values], dtype=np.int8))
    (artifacts_dir / "hidden_file_paths.json").write_text(json.dumps(file_paths, indent=2) + "\n", encoding="utf-8")

    print("2) Calcolo media e deviazione standard raw hidden sul solo train, per layer...")
    train = records["train"]
    sums = np.zeros((len(ROUTED_LAYERS), HIDDEN_DIMENSION), dtype=np.float64)
    squared_sums = np.zeros_like(sums)
    layer_counts = np.zeros(len(ROUTED_LAYERS), dtype=np.int64)
    reader = HiddenFileReader(file_paths)
    try:
        for start in range(0, len(train), args.batch_size):
            chunk = train[start : start + args.batch_size]
            indices = np.asarray([file_index[path] for path, _, _ in chunk], dtype=np.int32)
            offsets = np.asarray([offset for _, offset, _ in chunk], dtype=np.int64)
            layers = np.asarray([layer for _, _, layer in chunk], dtype=np.int8)
            vectors = reader.read_batch(indices, offsets).astype(np.float64)
            for layer in np.unique(layers):
                selected = vectors[layers == layer]
                sums[layer] += selected.sum(axis=0)
                squared_sums[layer] += np.square(selected).sum(axis=0)
                layer_counts[layer] += len(selected)
    finally:
        reader.close()
    observed_layers = layer_counts > 0
    # Build 6 may intentionally omit layer 3 from the current branch or layer
    # 42 from the preceding branch. Missing statistics are never indexed, but
    # remain finite to keep a self-contained 40-layer artifact.
    means = np.zeros_like(sums)
    means[observed_layers] = sums[observed_layers] / layer_counts[observed_layers, None]
    variances = np.zeros_like(sums)
    variances[observed_layers] = np.maximum(
        squared_sums[observed_layers] / layer_counts[observed_layers, None]
        - np.square(means[observed_layers]), 0.0
    )
    stds = np.sqrt(variances)
    stds[stds < 1e-6] = 1.0
    np.save(artifacts_dir / "hidden_train_mean_by_layer.npy", means.astype(np.float32))
    np.save(artifacts_dir / "hidden_train_std_by_layer.npy", stds.astype(np.float32))
    summary = {
        "experiment": "raw_hidden_projection_inputs",
        "raw_hidden_input": str(args.raw_hidden_input),
        "reference_path": args.reference_path,
        "layer_path": args.layer_path,
        "state_arrays_dir": str(args.state_arrays_dir),
        "raw_hidden_dimension": HIDDEN_DIMENSION,
        "routed_layers": list(ROUTED_LAYERS),
        "split_examples": counts,
        "raw_file_count": len(file_paths),
        "normalization": {
            "formula": "(h - train_mean[layer]) / train_std[layer]",
            "fit_split": "train",
        "per_layer": True,
        },
        "missing_train_layers": np.flatnonzero(~observed_layers).tolist(),
    }
    (artifacts_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nInput raw hidden preparati.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
