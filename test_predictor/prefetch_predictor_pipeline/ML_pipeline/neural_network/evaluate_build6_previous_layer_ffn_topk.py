#!/usr/bin/env python3
"""Evaluate the selected Build 6 FFN at Top-6, Top-10 and Top-20."""

from __future__ import annotations

import argparse
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from ffn_utils import (
    build_fixed_nnz_state_two_hidden_two_kv_ffn,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_materialized_kv_inputs,
    load_quantized_hidden_inputs,
    make_fixed_nnz_state_two_hidden_two_kv_dataset,
    require_tensorflow,
    save_json,
)


def parse_hidden_layers(value: str) -> list[int]:
    """Parse the comma-separated fusion layer list."""
    layers = [int(item) for item in value.split(",") if item.strip()]
    if any(item < 1 for item in layers):
        raise ValueError("--fusion-hidden-layers deve contenere solo interi positivi.")
    return layers


def calculate_recall_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Return the fraction of the six real experts contained in predicted Top-k."""
    if scores.shape != targets.shape or scores.shape[1] != 256:
        raise ValueError("Score e target devono avere shape [N, 256] identica.")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError("k non valido.")
    top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    rows = np.arange(scores.shape[0])[:, None]
    target_count = targets.sum(axis=1)
    if not np.all(target_count == 6):
        raise ValueError("Ogni esempio deve contenere esattamente sei expert target.")
    return float(targets[rows, top_indices].sum(axis=1).mean() / 6.0)


def calculate_target_rank_distribution(scores: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    """Count the rank of every real expert without materializing a full sort."""
    if scores.shape != targets.shape or scores.shape[1] != 256:
        raise ValueError("Score e target devono avere shape [N, 256] identica.")
    if not np.all(targets.sum(axis=1) == 6):
        raise ValueError("Ogni esempio deve contenere esattamente sei expert target.")

    target_expert_ids = np.argpartition(targets, -6, axis=1)[:, -6:]
    rank_counts = np.zeros(scores.shape[1] + 1, dtype=np.int64)
    rows_per_chunk = 8192
    for start in range(0, scores.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, scores.shape[0])
        chunk_scores = scores[start:end]
        chunk_target_ids = target_expert_ids[start:end]
        rows = np.arange(end - start)[:, None]
        target_scores = chunk_scores[rows, chunk_target_ids]
        ranks = 1 + (chunk_scores[:, :, None] > target_scores[:, None, :]).sum(axis=1)
        rank_counts += np.bincount(ranks.ravel(), minlength=scores.shape[1] + 1)

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
    """Load only model weights from a Keras archive; evaluation does not need Adam."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        weights_path = Path(temporary_directory) / "model.weights.h5"
        with zipfile.ZipFile(keras_path) as archive:
            with archive.open("model.weights.h5") as source, weights_path.open("wb") as destination:
                destination.write(source.read())
        model.load_weights(weights_path)


def parse_arguments() -> argparse.Namespace:
    """Read the selected Build 6 architecture and its local arrays."""
    parser = argparse.ArgumentParser(description="Valuta Recall@6, Recall@10, Recall@20 e Recall@50 della FFN Build 6.")
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--current-kv-data-dir", type=Path, required=True)
    parser.add_argument("--previous-kv-data-dir", type=Path, required=True)
    parser.add_argument("--token-position-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Modello .keras salvato da Colab.")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--state-hidden-units", type=int, required=True)
    parser.add_argument("--fusion-hidden-layers", required=True)
    parser.add_argument("--current-hidden-projection-dimension", type=int, required=True)
    parser.add_argument("--previous-hidden-projection-dimension", type=int, required=True)
    parser.add_argument("--current-kv-projection-dimension", choices=("direct", "64", "128", "256"), required=True)
    parser.add_argument("--previous-kv-projection-dimension", choices=("direct", "64", "128", "256"), required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def evaluate_split(model, args: argparse.Namespace, split: str, tf) -> dict[str, object]:
    """Predict one untouched split and calculate retrieval and rank metrics."""
    labels = load_labels(args.state_data_dir, split)
    indices, values, metadata = load_fixed_nnz_state_inputs(
        args.fixed_nnz_state_data_dir, split, labels.shape[0]
    )
    positions = np.load(
        args.token_position_data_dir / "arrays" / f"token_position_{split}.npy",
        mmap_mode="r",
    )
    current_hidden, current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, split, labels.shape[0]
    )
    previous_hidden, previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, split, labels.shape[0]
    )
    current_kv = load_materialized_kv_inputs(args.current_kv_data_dir, split, labels.shape[0])
    previous_kv = load_materialized_kv_inputs(args.previous_kv_data_dir, split, labels.shape[0])
    if positions.shape != (labels.shape[0], 1):
        raise ValueError(f"token_position {split} incompatibile.")
    dataset = make_fixed_nnz_state_two_hidden_two_kv_dataset(
        indices, values, labels, positions, current_hidden, previous_hidden,
        current_kv, previous_kv, args.batch_size, False, tf, args.batch_size,
    ).repeat()
    scores = model.predict(dataset, steps=math.ceil(labels.shape[0] / args.batch_size), verbose=0)
    rank_distribution = calculate_target_rank_distribution(scores, labels)
    return {
        "examples": int(labels.shape[0]),
        "recall_at_6": calculate_recall_at_k(scores, labels, 6),
        "recall_at_10": calculate_recall_at_k(scores, labels, 10),
        "recall_at_20": calculate_recall_at_k(scores, labels, 20),
        "recall_at_50": calculate_recall_at_k(scores, labels, 50),
        "target_rank_distribution": rank_distribution,
        "current_hidden_dequantization_scale": float(current_scale),
        "previous_hidden_dequantization_scale": float(previous_scale),
        "state_feature_count": int(metadata["feature_count"]),
    }


def main() -> None:
    """Rebuild candidate_09 and evaluate its saved best model without training."""
    args = parse_arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {args.checkpoint}")
    if args.output_report.exists():
        raise FileExistsError(f"Report gia' presente: {args.output_report}")
    if min(args.state_hidden_units, args.current_hidden_projection_dimension,
           args.previous_hidden_projection_dimension, args.batch_size) < 1:
        raise ValueError("Dimensioni o batch size non validi.")
    fusion_hidden_layers = parse_hidden_layers(args.fusion_hidden_layers)
    current_kv_projection = None if args.current_kv_projection_dimension == "direct" else int(args.current_kv_projection_dimension)
    previous_kv_projection = None if args.previous_kv_projection_dimension == "direct" else int(args.previous_kv_projection_dimension)

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    validation_labels = load_labels(args.state_data_dir, "validation")
    _, _, state_metadata = load_fixed_nnz_state_inputs(
        args.fixed_nnz_state_data_dir, "validation", validation_labels.shape[0]
    )
    _, current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, "validation", validation_labels.shape[0]
    )
    _, previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, "validation", validation_labels.shape[0]
    )
    print("1) Ricostruisco la candidata Build 6 e carico il modello migliore di Colab...")
    model = build_fixed_nnz_state_two_hidden_two_kv_ffn(
        state_metadata["feature_count"], state_metadata["input_slots"], args.state_hidden_units,
        fusion_hidden_layers, "relu", args.dropout, args.learning_rate, 0.0, tf,
        args.current_hidden_projection_dimension, args.previous_hidden_projection_dimension,
        current_scale, previous_scale, current_kv_projection, previous_kv_projection,
    )
    load_keras_weights(args.checkpoint, model)

    print("2) Calcolo Recall@6, Recall@10, Recall@20, Recall@50 e posizioni dei target su validation e test, senza addestrare...")
    validation = evaluate_split(model, args, "validation", tf)
    print(
        f"   validation: Recall@6={validation['recall_at_6']:.6f}; "
        f"Recall@10={validation['recall_at_10']:.6f}; Recall@20={validation['recall_at_20']:.6f}; "
        f"Recall@50={validation['recall_at_50']:.6f}"
    )
    print(
        "               target nelle posizioni 7-10="
        f"{1.0 - validation['recall_at_6'] - validation['target_rank_distribution']['beyond_top_10']['fraction_of_target_experts']:.6f}"
    )
    test = evaluate_split(model, args, "test", tf)
    print(
        f"   test:       Recall@6={test['recall_at_6']:.6f}; "
        f"Recall@10={test['recall_at_10']:.6f}; Recall@20={test['recall_at_20']:.6f}; "
        f"Recall@50={test['recall_at_50']:.6f}"
    )
    print(
        "               target nelle posizioni 7-10="
        f"{1.0 - test['recall_at_6'] - test['target_rank_distribution']['beyond_top_10']['fraction_of_target_experts']:.6f}"
    )
    save_json(args.output_report, {
        "experiment": "build6_previous_layer_ffn_topk_evaluation",
        "checkpoint": str(args.checkpoint),
        "selection": "candidate_09, best model from Colab epoch 23",
        "metrics_definition": {
            "recall_at_6": "Frazione dei sei expert target presenti nei sei score piu' alti.",
            "recall_at_10": "Frazione dei sei expert target presenti nei dieci score piu' alti.",
            "recall_at_20": "Frazione dei sei expert target presenti nei venti score piu' alti.",
            "recall_at_50": "Frazione dei sei expert target presenti nei cinquanta score piu' alti.",
            "target_rank_distribution": "Posizione esatta di ogni expert target nella classifica decrescente di 256 score.",
        },
        "validation": validation,
        "test": test,
    })
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
