#!/usr/bin/env python3
"""Evaluate the selected Build 7 Top-4 model at Top-6, 10, 20 and 50.

This program is deliberately read-only: it rebuilds the Top-4 architecture,
loads the saved weights, and evaluates the untouched validation and test
splits.  It never calls ``fit`` and never changes the checkpoint.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from ffn_utils import (
    build_fixed_nnz_state_two_hidden_gated_residual_ffn,
    load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_quantized_hidden_inputs,
    require_tensorflow,
    save_json,
)


def calculate_recall_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Return the fraction of the six real experts present in predicted Top-k."""
    if scores.shape != targets.shape or scores.shape[1] != 256:
        raise ValueError("Score e target devono avere shape [N, 256] identica.")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError("k non valido.")
    if not np.all(targets.sum(axis=1) == 6):
        raise ValueError("Ogni esempio deve contenere esattamente sei expert target.")
    top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    rows = np.arange(scores.shape[0])[:, None]
    return float(targets[rows, top_indices].sum(axis=1).mean() / 6.0)


def update_rank_counts(rank_counts: np.ndarray, scores: np.ndarray, targets: np.ndarray) -> None:
    """Accumulate exact ranks for a prediction batch into ``rank_counts``."""
    if scores.shape != targets.shape or scores.shape[1] != 256:
        raise ValueError("Score e target devono avere shape [N, 256] identica.")
    if not np.all(targets.sum(axis=1) == 6):
        raise ValueError("Ogni esempio deve contenere esattamente sei expert target.")

    target_expert_ids = np.argpartition(targets, -6, axis=1)[:, -6:]
    rows = np.arange(scores.shape[0])[:, None]
    target_scores = scores[rows, target_expert_ids]
    ranks = 1 + (scores[:, :, None] > target_scores[:, None, :]).sum(axis=1)
    rank_counts += np.bincount(ranks.ravel(), minlength=scores.shape[1] + 1)


def rank_distribution_from_counts(rank_counts: np.ndarray) -> dict[str, object]:
    """Serialize accumulated rank counts using the Build 6-compatible format."""

    total_targets = int(rank_counts.sum())
    counts_top_10 = [int(rank_counts[rank]) for rank in range(1, 11)]
    cumulative_top_10 = np.cumsum(counts_top_10)
    return {
        "target_experts": total_targets,
        "rank_1_to_10": [
            {
                "rank": rank,
                "count": count,
                "fraction_of_target_experts": float(count / total_targets),
            }
            for rank, count in enumerate(counts_top_10, start=1)
        ],
        "cumulative_recall_at_k_1_to_10": {
            str(rank): float(count / total_targets)
            for rank, count in enumerate(cumulative_top_10, start=1)
        },
        "beyond_top_10": {
            "count": int(rank_counts[11:].sum()),
            "fraction_of_target_experts": float(rank_counts[11:].sum() / total_targets),
        },
    }


def load_keras_weights(keras_path: Path, model) -> None:
    """Load only the model weights from a Keras archive, skipping Adam state."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        weights_path = Path(temporary_directory) / "model.weights.h5"
        with zipfile.ZipFile(keras_path) as archive:
            with archive.open("model.weights.h5") as source, weights_path.open("wb") as destination:
                destination.write(source.read())
        model.load_weights(weights_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valuta il migliore modello Build 7 Top-4 a Recall@6/10/20/50."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Modello Top-4 .keras da valutare.")
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--current-fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--splits", default="validation,test",
        help="Split separati da virgola da valutare: train, validation e/o test.",
    )
    parser.add_argument("--state-hidden-units", type=int, default=2048)
    parser.add_argument("--current-hidden-units", type=int, default=64)
    parser.add_argument("--previous-hidden-units", type=int, default=64)
    parser.add_argument("--fusion-hidden-units", type=int, default=2048)
    parser.add_argument("--l2-regularization", type=float, default=3e-6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def make_dataset(indices, values, labels, current_hidden, previous_hidden, transition, batch_size, tf):
    """Stream an untouched split without copying the full arrays."""
    examples, slots = indices.shape

    def batches():
        for start in range(0, examples, batch_size):
            rows = slice(start, min(start + batch_size, examples))
            yield {
                "state_feature_indices": indices[rows].astype(np.int32, copy=False),
                "state_feature_values": values[rows].astype(np.float32, copy=False),
                "raw_hidden_current": current_hidden[rows],
                "raw_hidden_previous": previous_hidden[rows],
                "expert_transition_probabilities": transition[rows].astype(np.float16, copy=False),
            }, labels[rows].astype(np.float32, copy=False)

    signature = ({
        "state_feature_indices": tf.TensorSpec((None, slots), tf.int32),
        "state_feature_values": tf.TensorSpec((None, slots), tf.float32),
        "raw_hidden_current": tf.TensorSpec((None, 4096), tf.int8),
        "raw_hidden_previous": tf.TensorSpec((None, 4096), tf.int8),
        "expert_transition_probabilities": tf.TensorSpec((None, 256), tf.float16),
    }, tf.TensorSpec((None, 256), tf.float32))
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def load_split(args: argparse.Namespace, split: str) -> dict[str, object]:
    labels = load_labels(args.state_data_dir, split)
    indices, values, metadata = load_fixed_nnz_state_inputs(
        args.current_fixed_nnz_state_data_dir, split, labels.shape[0]
    )
    current_hidden, current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, split, labels.shape[0]
    )
    previous_hidden, previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, split, labels.shape[0]
    )
    transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, split, labels.shape[0]
    )
    return {
        "labels": labels,
        "indices": indices,
        "values": values,
        "metadata": metadata,
        "current_hidden": current_hidden,
        "previous_hidden": previous_hidden,
        "transition": transition,
        "current_scale": current_scale,
        "previous_scale": previous_scale,
    }


def evaluate_split(model, data: dict[str, object], batch_size: int, tf) -> dict[str, object]:
    """Evaluate one split without materializing all prediction scores in RAM."""
    labels = data["labels"]
    dataset = make_dataset(
        data["indices"], data["values"], labels, data["current_hidden"],
        data["previous_hidden"], data["transition"], batch_size, tf,
    )
    recall_counts = {k: 0 for k in (6, 10, 20, 50)}
    rank_counts = np.zeros(257, dtype=np.int64)
    examples = 0
    for inputs, batch_targets in dataset:
        scores = np.asarray(model.predict_on_batch(inputs))
        batch_targets = np.asarray(batch_targets)
        rows = np.arange(scores.shape[0])[:, None]
        for k in recall_counts:
            top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
            recall_counts[k] += int(batch_targets[rows, top_indices].sum())
        update_rank_counts(rank_counts, scores, batch_targets)
        examples += scores.shape[0]
    if examples != labels.shape[0]:
        raise RuntimeError(f"Predizioni incomplete: {examples} esempi invece di {labels.shape[0]}.")
    return {
        "examples": int(examples),
        "recall_at_6": float(recall_counts[6] / (examples * 6)),
        "recall_at_10": float(recall_counts[10] / (examples * 6)),
        "recall_at_20": float(recall_counts[20] / (examples * 6)),
        "recall_at_50": float(recall_counts[50] / (examples * 6)),
        "target_rank_distribution": rank_distribution_from_counts(rank_counts),
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {args.checkpoint}")
    if args.output_report.exists():
        raise FileExistsError(f"Report gia' presente: {args.output_report}")
    if min(args.state_hidden_units, args.current_hidden_units, args.previous_hidden_units,
           args.fusion_hidden_units, args.batch_size) < 1:
        raise ValueError("Dimensioni o batch size non validi.")
    if args.l2_regularization < 0 or not 0 <= args.dropout < 1 or args.learning_rate <= 0:
        raise ValueError("Iperparametri non validi.")
    allowed_splits = {"train", "validation", "test"}
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits or len(set(splits)) != len(splits) or any(split not in allowed_splits for split in splits):
        raise ValueError("--splits deve contenere una o piu' voci distinte tra train, validation e test.")

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    print(f"1) Carico Build 7 Top-4 per: {', '.join(splits)}...")
    data_by_split = {split: load_split(args, split) for split in splits}
    reference_data = data_by_split[splits[0]]
    reference_metadata = reference_data["metadata"]
    for split, data in data_by_split.items():
        if data["metadata"] != reference_metadata:
            raise ValueError(f"Metadata state incompatibili per {split}.")
        if (data["current_scale"] != reference_data["current_scale"]
                or data["previous_scale"] != reference_data["previous_scale"]):
            raise ValueError(f"Scale int8 raw hidden incompatibili per {split}.")
    if reference_metadata != {"feature_count": 4411, "nonzero_features_per_example": 44, "input_slots": 45}:
        raise ValueError("Il valutatore Top-4 richiede lo state corrente Build 7 con 4411 feature e 45 slot.")
    print(
        f"   {splits[0]}: state={reference_data['indices'].shape}; "
        f"hidden={reference_data['current_hidden'].shape}/{reference_data['previous_hidden'].shape}; "
        f"q={reference_data['transition'].shape}"
    )
    print("2) Ricostruisco il checkpoint Top-4 e carico solo i pesi, senza addestrare...")
    model = build_fixed_nnz_state_two_hidden_gated_residual_ffn(
        reference_metadata["feature_count"], reference_metadata["input_slots"], args.state_hidden_units,
        args.current_hidden_units, args.previous_hidden_units,
        reference_data["current_scale"], reference_data["previous_scale"],
        args.fusion_hidden_units, args.dropout, args.learning_rate, args.l2_regularization, tf,
    )
    load_keras_weights(args.checkpoint, model)

    print("3) Calcolo Recall@6, @10, @20, @50 e posizioni dei target, batch per batch...")
    results = {split: evaluate_split(model, data_by_split[split], args.batch_size, tf) for split in splits}
    for split, result in results.items():
        print(
            f"   {split}: Recall@6={result['recall_at_6']:.6f}; "
            f"Recall@10={result['recall_at_10']:.6f}; Recall@20={result['recall_at_20']:.6f}; "
            f"Recall@50={result['recall_at_50']:.6f}"
        )

    save_json(args.output_report, {
        "experiment": "build7_top4_gated_residual_topk_evaluation",
        "checkpoint": str(args.checkpoint),
        "selection": "Build 7 Top-4 best checkpoint: state l + raw hidden l/l-1 + q gated residual",
        "architecture": {
            "state_fixed_nnz": "current layer only, 45 slots",
            "state_hidden_units": args.state_hidden_units,
            "current_hidden_projection_dimension": args.current_hidden_units,
            "previous_hidden_projection_dimension": args.previous_hidden_units,
            "fusion_hidden_units": args.fusion_hidden_units,
            "expert_transition": "direct_gated_residual",
        },
        "metrics_definition": {
            "recall_at_k": "Frazione dei sei expert target presenti nei k score piu' alti.",
            "target_rank_distribution": "Posizione esatta di ogni expert target nella classifica decrescente di 256 score.",
        },
        "splits_evaluated": splits,
        **results,
    })
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
