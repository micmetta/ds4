#!/usr/bin/env python3
"""Materialize leakage-safe expert-transition probabilities for Build 6.

For each routed layer, the script estimates P(next_expert | current_expert)
from consecutive-token routing labels.  The global train matrix is used by
validation/test; train rows use a deterministic prompt-grouped out-of-fold
matrix, so no row can consult a count originating from its own prompt fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


FIRST_ROUTED_LAYER = 3
NUM_ROUTED_LAYERS = 40
NUM_EXPERTS = 256
EXPERTS_PER_ROUTING = 6
SPLITS = ("train", "validation", "test")
DEFAULT_TRAIN_OOF_FOLDS = 5


def load_prompt_splits(path: Path) -> dict[int, str]:
    """Load the project prompt-level split assignment."""
    content = json.loads(path.read_text(encoding="utf-8"))
    assignments = content.get("split_per_prompt")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("Split per prompt mancante, vuoto o non valido.")
    result = {int(prompt_id): str(split) for prompt_id, split in assignments.items()}
    if any(split not in SPLITS for split in result.values()):
        raise ValueError("Lo split contiene un valore non supportato.")
    return result


def expected_counts(state_arrays_dir: Path) -> dict[str, int]:
    """Read row counts from the labels that define the canonical Build 6 order."""
    counts: dict[str, int] = {}
    for split in SPLITS:
        path = state_arrays_dir / f"y_{split}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Label non trovate: {path}")
        labels = np.load(path, mmap_mode="r")
        if labels.ndim != 2 or labels.shape[1] != NUM_EXPERTS:
            raise ValueError(f"{split}: label non valide in {path}")
        counts[split] = int(labels.shape[0])
    return counts


def iter_records(input_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield non-empty JSONL records with their one-based line number."""
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def validate_record(record: dict[str, Any], line_number: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return prompt, zero-based layer, current experts and next-token targets."""
    try:
        prompt_id = int(record["prompt_id"])
        layer = int(record["layer"])
        current = np.asarray(record["experts_t"], dtype=np.int16)
        target = np.asarray(record["target_experts_next_token"], dtype=np.int16)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Line {line_number}: record di routing incompleto.") from error

    if not FIRST_ROUTED_LAYER <= layer < FIRST_ROUTED_LAYER + NUM_ROUTED_LAYERS:
        raise ValueError(f"Line {line_number}: layer routed non valido: {layer}")
    for name, experts in (("experts_t", current), ("target_experts_next_token", target)):
        if experts.shape != (EXPERTS_PER_ROUTING,):
            raise ValueError(f"Line {line_number}: {name} deve contenere {EXPERTS_PER_ROUTING} expert.")
        if np.any(experts < 0) or np.any(experts >= NUM_EXPERTS):
            raise ValueError(f"Line {line_number}: {name} contiene un expert fuori intervallo.")
    return prompt_id, layer - FIRST_ROUTED_LAYER, current, target


def prompt_fold(prompt_id: int, folds: int) -> int:
    """Assign every prompt to one stable fold without using Python's salted hash."""
    return prompt_id % folds


def add_transition_counts(destination: np.ndarray, current: np.ndarray, target: np.ndarray) -> None:
    """Add the 6 x 6 current-to-next expert pairs to one layer count matrix."""
    for source_expert in current:
        np.add.at(destination[:, int(source_expert)], target, 1)


def probabilities_from_counts(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Apply additive Laplace smoothing along the target-expert axis."""
    numerator = counts.astype(np.float32) + np.float32(alpha)
    denominator = counts.sum(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    denominator += np.float32(NUM_EXPERTS * alpha)
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materializza P(next expert | current expert) e i vettori 256D Build 6."
    )
    parser.add_argument("--input", type=Path, required=True, help="Derived JSONL Build 6.")
    parser.add_argument("--split-by-prompt", type=Path, required=True)
    parser.add_argument("--state-arrays-dir", type=Path, required=True, help="Array label che fissano ordine e righe.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1, help="Smoothing additivo di Laplace.")
    parser.add_argument("--train-oof-folds", type=int, default=DEFAULT_TRAIN_OOF_FOLDS)
    args = parser.parse_args()

    if not args.input.is_file() or not args.split_by_prompt.is_file():
        raise FileNotFoundError("Input JSONL o split per prompt non trovato.")
    if args.output_dir.exists():
        raise FileExistsError(f"La directory di output esiste gia': {args.output_dir}")
    if args.alpha <= 0.0 or not np.isfinite(args.alpha):
        parser.error("--alpha deve essere finito e maggiore di zero.")
    if args.train_oof_folds < 2:
        parser.error("--train-oof-folds deve essere almeno 2.")

    assignments = load_prompt_splits(args.split_by_prompt)
    row_counts = expected_counts(args.state_arrays_dir)
    folds = args.train_oof_folds

    # Counts use [layer, target expert i, current expert j], matching P_l(i | j).
    global_train_counts = np.zeros((NUM_ROUTED_LAYERS, NUM_EXPERTS, NUM_EXPERTS), dtype=np.uint32)
    train_counts_by_fold = np.zeros((folds, NUM_ROUTED_LAYERS, NUM_EXPERTS, NUM_EXPERTS), dtype=np.uint32)
    observed_transitions = {split: 0 for split in SPLITS}

    print("1) Stimo i conteggi di transizione sul solo train e preparo i fold anti-leakage...")
    for line_number, record in iter_records(args.input):
        prompt_id, layer_index, current, target = validate_record(record, line_number)
        split = assignments.get(prompt_id)
        if split is None:
            raise ValueError(f"Line {line_number}: prompt {prompt_id} assente dallo split.")
        observed_transitions[split] += 1
        if split == "train":
            add_transition_counts(global_train_counts[layer_index], current, target)
            add_transition_counts(train_counts_by_fold[prompt_fold(prompt_id, folds), layer_index], current, target)

    if observed_transitions != row_counts:
        raise ValueError(f"Conteggi JSONL e array state non allineati: {observed_transitions} != {row_counts}")

    global_probabilities = probabilities_from_counts(global_train_counts, args.alpha)
    oof_probabilities = np.empty_like(train_counts_by_fold, dtype=np.float32)
    for fold in range(folds):
        # A train row in this fold consults all other train folds, never its own.
        oof_probabilities[fold] = probabilities_from_counts(
            global_train_counts.astype(np.int64) - train_counts_by_fold[fold].astype(np.int64), args.alpha
        )

    arrays_dir = args.output_dir / "arrays"
    artifacts_dir = args.output_dir / "artifacts"
    arrays_dir.mkdir(parents=True)
    artifacts_dir.mkdir()
    outputs = {
        split: np.lib.format.open_memmap(
            arrays_dir / f"expert_transition_probabilities_{split}.npy",
            mode="w+",
            dtype=np.float16,
            shape=(row_counts[split], NUM_EXPERTS),
        )
        for split in SPLITS
    }
    offsets = {split: 0 for split in SPLITS}

    print("2) Materializzo i vettori 256D float16 allineati agli array state...")
    for line_number, record in iter_records(args.input):
        prompt_id, layer_index, current, _ = validate_record(record, line_number)
        split = assignments.get(prompt_id)
        if split is None:
            raise ValueError(f"Line {line_number}: prompt {prompt_id} assente dallo split.")
        probabilities = (
            oof_probabilities[prompt_fold(prompt_id, folds), layer_index]
            if split == "train"
            else global_probabilities[layer_index]
        )
        # Mean the six conditional distributions P(:, j) for the current routing.
        position = offsets[split]
        outputs[split][position] = probabilities[:, current].mean(axis=1, dtype=np.float32).astype(np.float16)
        offsets[split] += 1

    if offsets != row_counts:
        raise RuntimeError(f"Scrittura incompleta: {offsets} != {row_counts}")
    for output in outputs.values():
        output.flush()

    np.save(artifacts_dir / "transition_probabilities_train_global_float32.npy", global_probabilities)
    for fold in range(folds):
        np.save(artifacts_dir / f"transition_probabilities_train_oof_fold_{fold:02d}_float32.npy", oof_probabilities[fold])

    summary = {
        "experiment": "build6_expert_transition_probabilities",
        "source_input": str(args.input),
        "split_by_prompt": str(args.split_by_prompt),
        "state_arrays_dir": str(args.state_arrays_dir),
        "probability_definition": "P_l(i|j), axes=[layer_index, target_expert_i, current_expert_j]",
        "routed_layers": list(range(FIRST_ROUTED_LAYER, FIRST_ROUTED_LAYER + NUM_ROUTED_LAYERS)),
        "num_experts": NUM_EXPERTS,
        "experts_per_routing": EXPERTS_PER_ROUTING,
        "alpha": args.alpha,
        "train_oof_folds": folds,
        "fold_assignment": "prompt_id modulo train_oof_folds",
        "anti_leakage": {
            "train": "out-of-fold per prompt_id; il fold del record e escluso dalla sua P",
            "validation_test": "P stimata su tutto il train",
        },
        "storage": {
            "probabilities": "float32",
            "probability_shape": [NUM_ROUTED_LAYERS, NUM_EXPERTS, NUM_EXPERTS],
            "feature_vectors": "float16",
            "feature_dimension": NUM_EXPERTS,
        },
        "split_examples": row_counts,
    }
    (artifacts_dir / "materialization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Feature di transizione completata.")
    print(f"Output: {args.output_dir}")
    print(f"   P globale float32 + {folds} P out-of-fold float32 in {artifacts_dir}")
    print(f"   Vettori 256D float16 in {arrays_dir}")


if __name__ == "__main__":
    main()
