#!/usr/bin/env python3
"""Measure which Build 6 input branches a trained FFN actually uses.

This is a post-training ablation evaluator.  It never changes model weights:
for one branch at a time it either replaces the input with zeroes or permutes
its examples deterministically.  Permutation preserves a branch's marginal
distribution while removing its relation with the current label.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path
from zlib import crc32

import numpy as np

from ffn_utils import (
    RANDOM_STATE,
    build_fixed_nnz_state_two_hidden_two_kv_ffn,
    calculate_recall_at_6,
    load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_materialized_kv_inputs,
    load_quantized_hidden_inputs,
    require_tensorflow,
    save_json,
)


BRANCHES = (
    "state_fixed_nnz",
    "raw_hidden_current",
    "raw_hidden_previous",
    "kv_current",
    "kv_previous",
    "token_position",
    "expert_transition_probabilities",
)


def load_keras_weights(keras_path: Path, model) -> None:
    """Load network weights from a .keras archive without restoring Adam."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        weights_path = Path(temporary_directory) / "model.weights.h5"
        with zipfile.ZipFile(keras_path) as archive:
            with archive.open("model.weights.h5") as source, weights_path.open("wb") as destination:
                destination.write(source.read())
        model.load_weights(weights_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablazione post-addestramento dei rami della FFN Build 6 con transition probabilities."
    )
    parser.add_argument("--tuning-report", type=Path, required=True)
    parser.add_argument("--candidate-id", default="candidate_10")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Modello .keras da valutare.")
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--current-kv-data-dir", type=Path, required=True)
    parser.add_argument("--previous-kv-data-dir", type=Path, required=True)
    parser.add_argument("--token-position-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--splits", default="validation,test", help="Split separati da virgola.")
    parser.add_argument("--modes", default="zero,permutation", help="zero, permutation, oppure entrambi.")
    parser.add_argument("--branches", default=",".join(BRANCHES), help="Rami separati da virgola.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fusion-batch-normalization", action="store_true")
    parser.add_argument(
        "--expert-transition-gated-residual",
        action="store_true",
        help="Ricostruisce il ramo q residuale gated invece della concatenazione diretta.",
    )
    return parser.parse_args()


def parse_csv(value: str, allowed: tuple[str, ...], argument: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(values).difference(allowed))
    if not values or unknown:
        raise ValueError(f"{argument} non valido. Valori ammessi: {', '.join(allowed)}.")
    return values


def select_candidate(report_path: Path, candidate_id: str) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    finalists = report.get("finalists_after_20_epochs", [])
    candidate = next((item for item in finalists if item["candidate_id"] == candidate_id), None)
    if candidate is None:
        available = ", ".join(item["candidate_id"] for item in finalists)
        raise ValueError(f"Candidata {candidate_id!r} assente. Disponibili: {available}.")
    return candidate


def load_split(args: argparse.Namespace, split: str) -> dict[str, np.ndarray | dict | float]:
    labels = load_labels(args.state_data_dir, split)
    indices, values, metadata = load_fixed_nnz_state_inputs(
        args.fixed_nnz_state_data_dir, split, labels.shape[0]
    )
    positions = np.load(args.token_position_data_dir / "arrays" / f"token_position_{split}.npy", mmap_mode="r")
    current_hidden, current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, split, labels.shape[0]
    )
    previous_hidden, previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, split, labels.shape[0]
    )
    current_kv = load_materialized_kv_inputs(args.current_kv_data_dir, split, labels.shape[0])
    previous_kv = load_materialized_kv_inputs(args.previous_kv_data_dir, split, labels.shape[0])
    transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, split, labels.shape[0]
    )
    return {
        "labels": labels, "indices": indices, "values": values, "metadata": metadata,
        "positions": positions, "current_hidden": current_hidden, "previous_hidden": previous_hidden,
        "current_kv": current_kv, "previous_kv": previous_kv, "transition": transition,
        "current_scale": current_scale, "previous_scale": previous_scale,
    }


def make_dataset(data: dict, batch_size: int, tf, branch: str | None = None, mode: str | None = None, seed: int = 0):
    """Stream one split; ablations allocate only a batch, never a full copy."""
    labels = data["labels"]
    examples = labels.shape[0]
    permutation = None
    if mode == "permutation":
        permutation = np.random.default_rng(seed).permutation(examples)

    branch_for_data = {
        "positions": "token_position",
        "current_hidden": "raw_hidden_current",
        "previous_hidden": "raw_hidden_previous",
        "current_kv": "kv_current",
        "previous_kv": "kv_previous",
        "transition": "expert_transition_probabilities",
    }

    def source(name: str, selected: slice):
        is_ablated = branch == branch_for_data[name]
        rows: slice | np.ndarray = selected
        if is_ablated and permutation is not None:
            rows = permutation[selected]
        values = data[name][rows]
        if is_ablated and mode == "zero":
            return np.zeros(values.shape, dtype=values.dtype)
        return values

    def batches():
        for start in range(0, examples, batch_size):
            selected = slice(start, min(start + batch_size, examples))
            if branch == "state_fixed_nnz" and permutation is not None:
                state_rows: slice | np.ndarray = permutation[selected]
            else:
                state_rows = selected
            state_indices = data["indices"][state_rows]
            state_values = data["values"][state_rows]
            if branch == "state_fixed_nnz" and mode == "zero":
                # Keeping ids is harmless: zero weights make the weighted embedding sum exactly zero.
                state_values = np.zeros(state_values.shape, dtype=state_values.dtype)
            yield {
                "state_feature_indices": state_indices.astype(np.int32, copy=False),
                "state_feature_values": state_values.astype(np.float32, copy=False),
                "token_position": source("positions", selected).astype(np.float32, copy=False),
                "raw_hidden_current": source("current_hidden", selected),
                "raw_hidden_previous": source("previous_hidden", selected),
                "kv_current_state": source("current_kv", selected).astype(np.float16, copy=False),
                "kv_previous_state": source("previous_kv", selected).astype(np.float16, copy=False),
                "expert_transition_probabilities": source("transition", selected).astype(np.float16, copy=False),
            }, labels[selected].astype(np.float32, copy=False)

    input_slots = data["indices"].shape[1]
    signature = (
        {
            "state_feature_indices": tf.TensorSpec((None, input_slots), tf.int32),
            "state_feature_values": tf.TensorSpec((None, input_slots), tf.float32),
            "token_position": tf.TensorSpec((None, 1), tf.float32),
            "raw_hidden_current": tf.TensorSpec((None, 4096), tf.int8),
            "raw_hidden_previous": tf.TensorSpec((None, 4096), tf.int8),
            "kv_current_state": tf.TensorSpec((None, 512), tf.float16),
            "kv_previous_state": tf.TensorSpec((None, 512), tf.float16),
            "expert_transition_probabilities": tf.TensorSpec((None, 256), tf.float16),
        }, tf.TensorSpec((None, 256), tf.float32),
    )
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def evaluate(model, data: dict, batch_size: int, tf, branch: str | None = None, mode: str | None = None, seed: int = 0) -> float:
    dataset = make_dataset(data, batch_size, tf, branch, mode, seed)
    scores = model.predict(dataset, steps=math.ceil(data["labels"].shape[0] / batch_size), verbose=0)
    return calculate_recall_at_6(scores, data["labels"])


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size deve essere positivo.")
    if not args.tuning_report.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("Tuning report o checkpoint .keras non trovato.")
    if args.output_report.exists():
        raise FileExistsError(f"Report gia' presente: {args.output_report}")
    splits = parse_csv(args.splits, ("train", "validation", "test"), "--splits")
    modes = parse_csv(args.modes, ("zero", "permutation"), "--modes")
    branches = parse_csv(args.branches, BRANCHES, "--branches")
    candidate = select_candidate(args.tuning_report, args.candidate_id)
    hp = candidate["hyperparameters"]
    current_kv = None if hp["current_kv_projection"] == "direct" else int(hp["current_kv_projection"])
    previous_kv = None if hp["previous_kv_projection"] == "direct" else int(hp["previous_kv_projection"])
    transition = 0 if hp["expert_transition_projection"] == "direct" else int(hp["expert_transition_projection"])
    fusion = [] if hp["fusion_hidden_units"] == 0 else [hp["fusion_hidden_units"]]

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    print("1) Carico gli split e ricostruisco la candidata selezionata...")
    data_by_split = {split: load_split(args, split) for split in splits}
    reference = data_by_split[splits[0]]
    model = build_fixed_nnz_state_two_hidden_two_kv_ffn(
        reference["metadata"]["feature_count"], reference["metadata"]["input_slots"],
        hp["state_hidden_units"], fusion, "relu", 0.3, hp["learning_rate"], 0.0, tf,
        hp["current_raw_hidden_projection_dimension"], hp["previous_raw_hidden_projection_dimension"],
        reference["current_scale"], reference["previous_scale"], current_kv, previous_kv,
        ranking_loss_weight=0.0, ranking_loss_margin=0.05,
        expert_transition_projection_dimension=transition,
        fusion_batch_normalization=args.fusion_batch_normalization,
        expert_transition_gated_residual=args.expert_transition_gated_residual,
    )
    load_keras_weights(args.checkpoint, model)
    print("2) Calcolo baseline e ablazioni: zero e permutazione per un ramo alla volta...")
    results: dict[str, dict] = {}
    for split in splits:
        data = data_by_split[split]
        baseline = evaluate(model, data, args.batch_size, tf)
        split_results: dict[str, object] = {
            "examples": int(data["labels"].shape[0]), "baseline_recall_at_6": baseline, "ablations": {}
        }
        print(f"   {split}: baseline Recall@6={baseline:.6f}")
        for branch in branches:
            branch_results = {}
            for mode in modes:
                seed = RANDOM_STATE + crc32(f"{split}:{branch}".encode())
                recall = evaluate(model, data, args.batch_size, tf, branch, mode, seed)
                branch_results[mode] = {
                    "recall_at_6": recall,
                    "drop_from_baseline": baseline - recall,
                }
                print(f"      {branch} [{mode}]: Recall@6={recall:.6f}; delta={baseline - recall:+.6f}")
            split_results["ablations"][branch] = branch_results
        results[split] = split_results
    save_json(args.output_report, {
        "experiment": "build6_neural_network_post_training_branch_ablation",
        "checkpoint": str(args.checkpoint), "tuning_report": str(args.tuning_report),
        "candidate_id": args.candidate_id,
        "fusion_batch_normalization": args.fusion_batch_normalization,
        "expert_transition_gated_residual": args.expert_transition_gated_residual,
        "methods": {
            "zero": "Sostituisce il ramo con zero; per lo state fixed-NNZ azzera solo i valori, rendendo nullo il weighted embedding sum.",
            "permutation": "Permuta deterministicamente le righe del singolo ramo, preservandone distribuzione e shape ma rompendo l'allineamento con label e altri rami.",
            "drop_from_baseline": "Recall@6 baseline meno Recall@6 dopo ablazione: positivo significa che il ramo aiuta questo modello addestrato.",
        },
        "results": results,
    })
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
