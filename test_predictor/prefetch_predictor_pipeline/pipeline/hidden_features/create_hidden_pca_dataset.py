#!/usr/bin/env python3
"""Create layer-wise train-only hidden PCA without loading the full JSONL in RAM."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np
from sklearn.decomposition import IncrementalPCA


HIDDEN_DIMENSION = 4096
DEFAULT_COMPONENTS = 64
DEFAULT_BATCH_SIZE = 4096
ROUTED_LAYERS = tuple(range(3, 43))


def load_split_per_prompt(path: Path) -> dict[int, str]:
    """Load the fixed prompt-level train/validation/test split."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("split_per_prompt")
    if not isinstance(mapping, dict):
        raise ValueError(f"{path} does not contain split_per_prompt.")
    return {int(prompt_id): str(split) for prompt_id, split in mapping.items()}


def validate_reference(reference: object, line_number: int) -> dict[str, Any]:
    """Validate and return one raw hidden-vector reference."""
    if not isinstance(reference, dict):
        raise ValueError(f"Line {line_number}: missing hidden_raw_ref.")
    if reference.get("dtype") != "float32" or reference.get("shape") != [HIDDEN_DIMENSION]:
        raise ValueError(f"Line {line_number}: expected float32[{HIDDEN_DIMENSION}] hidden vector.")
    return reference


class HiddenReader:
    """Read fixed-size float32 hidden vectors from collector binary files."""

    def __init__(self) -> None:
        self.files: dict[str, BinaryIO] = {}

    def read(self, reference: dict[str, Any]) -> np.ndarray:
        path = str(reference["file"])
        file = self.files.get(path)
        if file is None:
            file = Path(path).open("rb")
            self.files[path] = file
        file.seek(int(reference["offset_bytes"]))
        vector = np.fromfile(file, dtype=np.float32, count=HIDDEN_DIMENSION)
        if vector.shape != (HIDDEN_DIMENSION,):
            raise ValueError(f"Incomplete hidden vector at {path}:{reference['offset_bytes']}.")
        return vector

    def close(self) -> None:
        for file in self.files.values():
            file.close()
        self.files.clear()


def shard_references(input_path: Path, split_per_prompt: dict[int, str], work_dir: Path) -> tuple[dict[int, Path], int]:
    """Write compact per-layer references, preserving the original row order."""
    shard_dir = work_dir / "references_by_layer"
    shard_dir.mkdir(parents=True)
    paths = {layer: shard_dir / f"layer_{layer:02d}.jsonl" for layer in ROUTED_LAYERS}
    handles = {layer: paths[layer].open("w", encoding="utf-8") for layer in ROUTED_LAYERS}
    rows = 0
    try:
        with input_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                record = json.loads(line)
                layer = int(record["layer"])
                prompt_id = int(record["prompt_id"])
                if layer not in handles:
                    raise ValueError(f"Line {line_number}: unsupported routed layer {layer}.")
                split = split_per_prompt.get(prompt_id)
                if split not in {"train", "validation", "test"}:
                    raise ValueError(f"Line {line_number}: no valid split for prompt {prompt_id}.")
                reference = validate_reference(record.get("hidden_raw_ref"), line_number)
                json.dump({"row": rows, "split": split, "reference": reference}, handles[layer], separators=(",", ":"))
                handles[layer].write("\n")
                rows += 1
    finally:
        for handle in handles.values():
            handle.close()
    return paths, rows


def load_shard(path: Path) -> list[dict[str, Any]]:
    """Load one small per-layer reference shard into memory."""
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def reference_batches(entries: list[dict[str, Any]], batch_size: int, reader: HiddenReader) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield record rows and raw hidden vectors in bounded float32 batches."""
    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        matrix = np.empty((len(chunk), HIDDEN_DIMENSION), dtype=np.float32)
        for index, entry in enumerate(chunk):
            matrix[index] = reader.read(entry["reference"])
        yield np.asarray([entry["row"] for entry in chunk], dtype=np.int64), matrix


def fit_and_transform_layer(
    layer: int,
    entries: list[dict[str, Any]],
    components: int,
    batch_size: int,
    artifacts_dir: Path,
    projection_values: np.memmap,
    reader: HiddenReader,
) -> None:
    """Fit one incremental PCA on train entries and transform all layer entries."""
    train_entries = [entry for entry in entries if entry["split"] == "train"]
    if len(train_entries) < components:
        raise ValueError(f"Layer {layer} has {len(train_entries)} train rows; need at least {components}.")
    print(f"PCA layer {layer}: fit on {len(train_entries)} train vectors in batches of {batch_size}.")
    pca = IncrementalPCA(n_components=components, batch_size=batch_size)
    for _, matrix in reference_batches(train_entries, batch_size, reader):
        pca.partial_fit(matrix)
    np.savez_compressed(
        artifacts_dir / f"hidden_pca_layer_{layer:02d}.npz",
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        algorithm="IncrementalPCA",
        batch_size=batch_size,
    )
    for rows, matrix in reference_batches(entries, batch_size, reader):
        projection_values[rows] = pca.transform(matrix).astype(np.float32)


def write_records(input_path: Path, output_path: Path, projection_values: np.memmap, components: int) -> None:
    """Stream the original JSONL and replace raw references with PCA values."""
    projection_id = f"incremental_pca_layerwise_{components}_train_v1"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as destination:
        for row, line in enumerate(source):
            record = json.loads(line)
            validate_reference(record.get("hidden_raw_ref"), row + 1)
            record["hidden_features"] = {
                "source_tensor": "ffn_norm", "projection": "pca", "projection_id": projection_id,
                "input_dimension": HIDDEN_DIMENSION, "output_dimension": components,
                "values": projection_values[row].astype(float).tolist(),
            }
            del record["hidden_raw_ref"]
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add memory-bounded layer-wise PCA hidden features to a dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-by-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.components < 1 or args.components > HIDDEN_DIMENSION or args.batch_size < args.components:
        parser.error("--components deve essere in 1..4096 e --batch-size deve essere almeno --components.")
    if not args.input.is_file() or not args.split_by_prompt.is_file():
        raise FileNotFoundError("Input JSONL o split_by_prompt non trovato.")
    if args.output.exists() or args.artifacts_dir.exists():
        raise FileExistsError("Output o artifacts esistono gia': rimuovi gli artefatti parziali o scegli nuovi percorsi.")

    work_dir = args.artifacts_dir.parent / f".{args.artifacts_dir.name}_work"
    if work_dir.exists():
        raise FileExistsError(f"Directory temporanea esistente: {work_dir}")
    work_dir.mkdir(parents=True)
    reader = HiddenReader()
    try:
        print("1) Indicizzo le reference raw per layer senza caricare il JSONL completo...")
        shards, record_count = shard_references(args.input, load_split_per_prompt(args.split_by_prompt), work_dir)
        args.artifacts_dir.mkdir()
        projection_path = work_dir / "hidden_pca_values.float32.memmap"
        projection_values = np.memmap(projection_path, mode="w+", dtype=np.float32, shape=(record_count, args.components))
        print("2) Fit PCA train-only e trasformazione per layer, in batch...")
        for layer in ROUTED_LAYERS:
            fit_and_transform_layer(layer, load_shard(shards[layer]), args.components, args.batch_size, args.artifacts_dir, projection_values, reader)
        projection_values.flush()
        print("3) Scrivo il JSONL PCA in streaming...")
        write_records(args.input, args.output, projection_values, args.components)
        print(f"Hidden PCA dataset complete: {record_count} examples -> {args.output}")
    finally:
        reader.close()
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
