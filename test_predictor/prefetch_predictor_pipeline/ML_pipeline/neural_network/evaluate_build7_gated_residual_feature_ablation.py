#!/usr/bin/env python3
"""Measure branch and state-block importance for the selected Build 7 FFN.

The evaluator is read-only: weights are never changed.  It measures the drop
in Recall@6 when one complete branch, or one of the causal state blocks l,
l-1 or l-2, is zeroed or deterministically permuted.  Defaults deliberately
exclude test: model selection and diagnostics use only train and validation.
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
    "state_current_fixed_nnz",
    "state_previous_fixed_nnz",
    "state_previous_previous_fixed_nnz",
    "raw_hidden_current",
    "raw_hidden_previous",
    "kv_current",
    "kv_previous",
    "token_position",
    "expert_transition_probabilities",
)
STATE_BRANCHES = set(BRANCHES[:4])


def load_keras_weights(keras_path: Path, model) -> None:
    """Load only model weights from a Keras archive, intentionally skipping Adam."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        weights_path = Path(temporary_directory) / "model.weights.h5"
        with zipfile.ZipFile(keras_path) as archive:
            with archive.open("model.weights.h5") as source, weights_path.open("wb") as destination:
                destination.write(source.read())
        model.load_weights(weights_path)


def parse_csv(value: str, allowed: tuple[str, ...], argument: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(values).difference(allowed))
    if not values or unknown:
        raise ValueError(f"{argument} non valido. Valori ammessi: {', '.join(allowed)}.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablazione post-training Build 7: rami FFN e blocchi state l/l-1/l-2."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Modello .keras selezionato.")
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument(
        "--state-feature-names", type=Path, required=True,
        help="feature_names.json del preprocessing Build 7, usato per separare l, l-1 e l-2.",
    )
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--current-kv-data-dir", type=Path, required=True)
    parser.add_argument("--previous-kv-data-dir", type=Path, required=True)
    parser.add_argument("--token-position-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--modes", default="zero,permutation")
    parser.add_argument("--branches", default=",".join(BRANCHES))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--state-hidden-units", type=int, default=2048)
    parser.add_argument("--l2-regularization", type=float, default=3e-6)
    parser.add_argument("--branch-dropout", type=float, default=0.0)
    return parser.parse_args()


def state_component_masks(feature_names_path: Path, feature_count: int) -> dict[str, np.ndarray]:
    payload = json.loads(feature_names_path.read_text(encoding="utf-8"))
    names = payload.get("feature_names")
    if not isinstance(names, list) or len(names) != feature_count:
        raise ValueError("feature_names.json incompatibile con feature_count dello state fixed-NNZ.")
    names_array = np.asarray(names, dtype=str)
    previous_previous = np.char.startswith(names_array, "previous_previous_")
    previous = np.char.startswith(names_array, "previous_") & ~previous_previous
    current = ~(previous | previous_previous)
    masks = {
        "state_current_fixed_nnz": current,
        "state_previous_fixed_nnz": previous,
        "state_previous_previous_fixed_nnz": previous_previous,
    }
    expected = {"state_current_fixed_nnz": 4411, "state_previous_fixed_nnz": 4373,
                "state_previous_previous_fixed_nnz": 4373}
    for name, mask in masks.items():
        if int(mask.sum()) != expected[name]:
            raise ValueError(f"{name}: feature trovate {int(mask.sum())}, attese {expected[name]}.")
    # The final fixed-NNZ slot is the bias sentinel at index feature_count.
    # Append false so it is never considered part of a causal state block.
    return {name: np.concatenate([mask, np.array([False])]) for name, mask in masks.items()}


def load_split(args: argparse.Namespace, split: str) -> dict[str, object]:
    labels = load_labels(args.state_data_dir, split)
    indices, values, metadata = load_fixed_nnz_state_inputs(
        args.fixed_nnz_state_data_dir, split, labels.shape[0]
    )
    position_path = args.token_position_data_dir / "arrays" / f"token_position_{split}.npy"
    positions = np.load(position_path, mmap_mode="r")
    if positions.shape != (labels.shape[0], 1) or positions.dtype != np.float32:
        raise ValueError(f"token_position incompatibile: {position_path}")
    current_hidden, current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, split, labels.shape[0]
    )
    previous_hidden, previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, split, labels.shape[0]
    )
    return {
        "labels": labels,
        "indices": indices,
        "values": values,
        "metadata": metadata,
        "positions": positions,
        "current_hidden": current_hidden,
        "previous_hidden": previous_hidden,
        "current_kv": load_materialized_kv_inputs(args.current_kv_data_dir, split, labels.shape[0]),
        "previous_kv": load_materialized_kv_inputs(args.previous_kv_data_dir, split, labels.shape[0]),
        "transition": load_expert_transition_probabilities_inputs(
            args.expert_transition_probabilities_data_dir, split, labels.shape[0]
        ),
        "current_scale": current_scale,
        "previous_scale": previous_scale,
    }


def make_dataset(
    data: dict[str, object], batch_size: int, tf, state_masks: dict[str, np.ndarray],
    branch: str | None = None, mode: str | None = None, seed: int = 0,
):
    """Create streamed ablated batches without materialising a second split."""
    labels = data["labels"]
    examples = labels.shape[0]
    permutation = np.random.default_rng(seed).permutation(examples) if mode == "permutation" else None
    data_branch = {
        "positions": "token_position",
        "current_hidden": "raw_hidden_current",
        "previous_hidden": "raw_hidden_previous",
        "current_kv": "kv_current",
        "previous_kv": "kv_previous",
        "transition": "expert_transition_probabilities",
    }

    def source(name: str, selected: slice):
        rows: slice | np.ndarray = selected
        if branch == data_branch[name] and permutation is not None:
            rows = permutation[selected]
        values = data[name][rows]
        if branch == data_branch[name] and mode == "zero":
            return np.zeros(values.shape, dtype=values.dtype)
        return values

    def state_inputs(selected: slice):
        state_rows: slice | np.ndarray = selected
        if branch == "state_fixed_nnz" and permutation is not None:
            state_rows = permutation[selected]
        indices = np.asarray(data["indices"][state_rows])
        values = np.asarray(data["values"][state_rows])
        if branch == "state_fixed_nnz" and mode == "zero":
            values = np.zeros(values.shape, dtype=values.dtype)
        if branch in state_masks:
            selected_mask = state_masks[branch][indices]
            if mode == "zero":
                values = values.copy()
                values[selected_mask] = 0.0
            elif permutation is not None:
                source_indices = np.asarray(data["indices"][permutation[selected]])
                source_values = np.asarray(data["values"][permutation[selected]])
                source_mask = state_masks[branch][source_indices]
                if not np.array_equal(selected_mask, source_mask):
                    raise ValueError(f"{branch}: slot fixed-NNZ non allineati tra le righe.")
                indices = indices.copy()
                values = values.copy()
                indices[selected_mask] = source_indices[selected_mask]
                values[selected_mask] = source_values[selected_mask]
        return indices, values

    def batches():
        for start in range(0, examples, batch_size):
            selected = slice(start, min(start + batch_size, examples))
            state_indices, state_values = state_inputs(selected)
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

    slots = data["indices"].shape[1]
    signature = ({
        "state_feature_indices": tf.TensorSpec((None, slots), tf.int32),
        "state_feature_values": tf.TensorSpec((None, slots), tf.float32),
        "token_position": tf.TensorSpec((None, 1), tf.float32),
        "raw_hidden_current": tf.TensorSpec((None, 4096), tf.int8),
        "raw_hidden_previous": tf.TensorSpec((None, 4096), tf.int8),
        "kv_current_state": tf.TensorSpec((None, 512), tf.float16),
        "kv_previous_state": tf.TensorSpec((None, 512), tf.float16),
        "expert_transition_probabilities": tf.TensorSpec((None, 256), tf.float16),
    }, tf.TensorSpec((None, 256), tf.float32))
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def evaluate(model, data, batch_size, tf, state_masks, branch=None, mode=None, seed=0) -> float:
    dataset = make_dataset(data, batch_size, tf, state_masks, branch, mode, seed)
    scores = model.predict(dataset, steps=math.ceil(data["labels"].shape[0] / batch_size), verbose=0)
    return calculate_recall_at_6(scores, data["labels"])


def main() -> None:
    args = parse_args()
    if (args.batch_size < 1 or args.state_hidden_units < 1 or args.l2_regularization < 0
            or args.branch_dropout < 0 or args.branch_dropout >= 1):
        raise ValueError("Parametri di valutazione non validi.")
    if not args.checkpoint.is_file() or not args.state_feature_names.is_file():
        raise FileNotFoundError("Checkpoint .keras o feature_names.json non trovato.")
    if args.output_report.exists():
        raise FileExistsError(f"Report gia' presente: {args.output_report}")
    splits = parse_csv(args.splits, ("train", "validation", "test"), "--splits")
    modes = parse_csv(args.modes, ("zero", "permutation"), "--modes")
    branches = parse_csv(args.branches, BRANCHES, "--branches")

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    print("1) Carico train/validation Build 7 e separo i blocchi state l/l-1/l-2...")
    data_by_split = {split: load_split(args, split) for split in splits}
    reference = data_by_split[splits[0]]
    state_masks = state_component_masks(args.state_feature_names, reference["metadata"]["feature_count"])
    for split, data in data_by_split.items():
        if data["metadata"] != reference["metadata"]:
            raise ValueError(f"{split}: metadata state incompatibili.")
        if data["current_scale"] != reference["current_scale"] or data["previous_scale"] != reference["previous_scale"]:
            raise ValueError(f"{split}: scale int8 hidden incompatibili.")

    print("2) Ricostruisco il migliore modello Build 7 (state 2048, L2=3e-6, q gated residual)...")
    model = build_fixed_nnz_state_two_hidden_two_kv_ffn(
        reference["metadata"]["feature_count"], reference["metadata"]["input_slots"],
        args.state_hidden_units, [2048], "relu", 0.3, 0.00015, args.l2_regularization, tf,
        64, 64, reference["current_scale"], reference["previous_scale"], 256, None,
        ranking_loss_weight=0.0, ranking_loss_margin=0.05,
        expert_transition_projection_dimension=0,
        expert_transition_gated_residual=True,
        branch_dropout=args.branch_dropout,
    )
    load_keras_weights(args.checkpoint, model)

    print("3) Calcolo baseline e ablazioni per un ramo/blocco alla volta...")
    results: dict[str, object] = {}
    for split in splits:
        data = data_by_split[split]
        baseline = evaluate(model, data, args.batch_size, tf, state_masks)
        ablations: dict[str, object] = {}
        print(f"   {split}: baseline Recall@6={baseline:.6f}")
        for branch in branches:
            branch_results: dict[str, object] = {}
            for mode in modes:
                seed = RANDOM_STATE + crc32(f"{split}:{branch}".encode())
                recall = evaluate(model, data, args.batch_size, tf, state_masks, branch, mode, seed)
                drop = baseline - recall
                branch_results[mode] = {"recall_at_6": recall, "drop_from_baseline": drop}
                print(f"      {branch} [{mode}]: Recall@6={recall:.6f}; delta={drop:+.6f}")
            ablations[branch] = branch_results
        results[split] = {
            "examples": int(data["labels"].shape[0]),
            "baseline_recall_at_6": baseline,
            "ablations": ablations,
        }
    save_json(args.output_report, {
        "experiment": "build7_gated_residual_post_training_feature_ablation",
        "checkpoint": str(args.checkpoint),
        "splits": splits,
        "architecture": {
            "state_hidden_units": args.state_hidden_units,
            "l2_regularization": args.l2_regularization,
            "branch_dropout": args.branch_dropout,
            "expert_transition": "direct_gated_residual",
        },
        "state_blocks": {
            "state_current_fixed_nnz": "Feature del layer target-context l.",
            "state_previous_fixed_nnz": "Feature tabellari del layer l-1.",
            "state_previous_previous_fixed_nnz": "Nuove feature tabellari del layer l-2 introdotte da Build 7.",
        },
        "methods": {
            "zero": "Azzera il ramo; per i blocchi state azzera solo i loro valori mantenendo gli slot fixed-NNZ.",
            "permutation": "Permuta deterministicamente le righe del solo ramo/blocco, preservandone distribuzione e shape ma rompendo l'allineamento.",
            "drop_from_baseline": "Recall@6 baseline meno Recall@6 dopo ablazione; positivo indica che il ramo aiuta il modello.",
        },
        "results": results,
    })
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
