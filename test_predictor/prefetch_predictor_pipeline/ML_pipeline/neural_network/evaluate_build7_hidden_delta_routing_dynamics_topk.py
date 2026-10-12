#!/usr/bin/env python3
"""Evaluate the AutoML Build 7 hidden-delta + routing model at multiple Top-k values.

The architecture is read from the AutoML ``final_ranking.json`` report, so the
checkpoint is rebuilt with exactly the dimensions used by the winning trial.
This program is read-only: it never trains and it never overwrites weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ffn_utils import (
    build_fixed_nnz_state_two_hidden_gated_residual_two_candidate_ffn,
    load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_quantized_hidden_inputs,
    require_tensorflow,
    save_json,
)
from train_build7_routing_dynamics_hidden_delta import candidates, dataset


TOP_K_VALUES = (6, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-report", type=Path, required=True)
    parser.add_argument("--candidate-features-dir", type=Path, required=True)
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--current-fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def load_configuration(report_path: Path) -> tuple[str, dict[str, object]]:
    """Extract the chosen trial and its architecture from AutoML final ranking."""
    with report_path.open(encoding="utf-8") as source:
        report = json.load(source)
    winner = report.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("Il report AutoML non contiene il dizionario 'winner'.")
    trial_id = winner.get("trial_id")
    configuration = winner.get("configuration")
    if not isinstance(trial_id, str) or not isinstance(configuration, dict):
        raise ValueError("Il vincitore AutoML non contiene trial_id/configuration validi.")
    required = {
        "state_hidden_units", "current_hidden_projection_dimension",
        "previous_hidden_projection_dimension", "fusion_hidden_units", "dropout",
        "learning_rate", "l2_regularization", "hidden_delta_units", "routing_dynamics_units",
    }
    missing = sorted(required - set(configuration))
    if missing:
        raise ValueError(f"Configurazione AutoML incompleta: {', '.join(missing)}")
    return trial_id, configuration


def update_rank_counts(rank_counts: np.ndarray, scores: np.ndarray, targets: np.ndarray) -> None:
    target_ids = np.argpartition(targets, -6, axis=1)[:, -6:]
    rows = np.arange(scores.shape[0])[:, None]
    target_scores = scores[rows, target_ids]
    ranks = 1 + (scores[:, :, None] > target_scores[:, None, :]).sum(axis=1)
    rank_counts += np.bincount(ranks.ravel(), minlength=257)


def rank_distribution(rank_counts: np.ndarray) -> dict[str, object]:
    total = int(rank_counts.sum())
    counts = [int(rank_counts[rank]) for rank in range(1, 11)]
    cumulative = np.cumsum(counts)
    return {
        "target_experts": total,
        "rank_1_to_10": [
            {"rank": rank, "count": count, "fraction_of_target_experts": float(count / total)}
            for rank, count in enumerate(counts, start=1)
        ],
        "cumulative_recall_at_k_1_to_10": {
            str(rank): float(value / total) for rank, value in enumerate(cumulative, start=1)
        },
        "beyond_top_10": {
            "count": int(rank_counts[11:].sum()),
            "fraction_of_target_experts": float(rank_counts[11:].sum() / total),
        },
    }


def load_split(args: argparse.Namespace, split: str) -> dict[str, object]:
    labels = load_labels(args.state_data_dir, split)
    indices, values, metadata = load_fixed_nnz_state_inputs(
        args.current_fixed_nnz_state_data_dir, split, labels.shape[0]
    )
    current, current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, split, labels.shape[0]
    )
    previous, previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, split, labels.shape[0]
    )
    transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, split, labels.shape[0]
    )
    delta, delta_aux, routing, delta_scale = candidates(
        args.candidate_features_dir, split, labels.shape[0]
    )
    return {
        "labels": labels, "indices": indices, "values": values, "metadata": metadata,
        "current": current, "current_scale": current_scale,
        "previous": previous, "previous_scale": previous_scale,
        "transition": transition, "delta": delta, "delta_aux": delta_aux,
        "routing": routing, "delta_scale": delta_scale,
    }


def evaluate_split(model, data: dict[str, object], batch_size: int, tf) -> dict[str, object]:
    labels = data["labels"]
    input_data = dataset(
        data["indices"], data["values"], labels, data["current"], data["previous"],
        data["transition"], data["delta"], data["delta_aux"], data["routing"],
        batch_size, False, tf,
    )
    counts = {k: 0 for k in TOP_K_VALUES}
    rank_counts = np.zeros(257, dtype=np.int64)
    seen = 0
    for inputs, targets in input_data:
        scores = np.asarray(model.predict_on_batch(inputs))
        targets = np.asarray(targets)
        rows = np.arange(scores.shape[0])[:, None]
        for k in TOP_K_VALUES:
            top_ids = np.argpartition(scores, -k, axis=1)[:, -k:]
            counts[k] += int(targets[rows, top_ids].sum())
        update_rank_counts(rank_counts, scores, targets)
        seen += scores.shape[0]
    if seen != labels.shape[0]:
        raise RuntimeError(f"Predizioni incomplete: {seen} invece di {labels.shape[0]} esempi.")
    return {
        "examples": int(seen),
        **{f"recall_at_{k}": float(counts[k] / (seen * 6)) for k in TOP_K_VALUES},
        "target_rank_distribution": rank_distribution(rank_counts),
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file() or not args.architecture_report.is_file():
        raise FileNotFoundError("Checkpoint o report di architettura AutoML non trovato.")
    if args.output_report.exists():
        raise FileExistsError(f"Report gia' presente: {args.output_report}")
    if args.batch_size < 1:
        raise ValueError("--batch-size deve essere positivo.")
    allowed = {"train", "validation", "test"}
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits or len(splits) != len(set(splits)) or any(item not in allowed for item in splits):
        raise ValueError("--splits deve contenere voci distinte tra train, validation e test.")

    trial_id, configuration = load_configuration(args.architecture_report)
    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    print(f"1) Carico il vincitore AutoML {trial_id} per: {', '.join(splits)}...")
    print(f"   configurazione={configuration}")
    data_by_split = {split: load_split(args, split) for split in splits}
    reference = data_by_split[splits[0]]
    metadata = reference["metadata"]
    for split, data in data_by_split.items():
        if data["metadata"] != metadata or data["current_scale"] != reference["current_scale"] or data["previous_scale"] != reference["previous_scale"] or data["delta_scale"] != reference["delta_scale"]:
            raise ValueError(f"Metadata o scale incompatibili nello split {split}.")

    print("2) Ricostruisco l'architettura AutoML e carico i pesi senza addestrare...")
    model = build_fixed_nnz_state_two_hidden_gated_residual_two_candidate_ffn(
        metadata["feature_count"], metadata["input_slots"],
        int(configuration["state_hidden_units"]),
        int(configuration["current_hidden_projection_dimension"]),
        int(configuration["previous_hidden_projection_dimension"]),
        reference["current_scale"], reference["previous_scale"],
        int(configuration["fusion_hidden_units"]), float(configuration["dropout"]),
        float(configuration["learning_rate"]), float(configuration["l2_regularization"]),
        4096, int(configuration["hidden_delta_units"]), "int8", reference["delta_scale"],
        4, 261, int(configuration["routing_dynamics_units"]), "float16", tf,
    )
    model.load_weights(args.checkpoint)

    print("3) Calcolo Recall@6/@10/@20/@50 e distribuzione dei rank...")
    results = {split: evaluate_split(model, data, args.batch_size, tf) for split, data in data_by_split.items()}
    for split, result in results.items():
        print("   " + split + ": " + "; ".join(
            f"Recall@{k}={result[f'recall_at_{k}']:.6f}" for k in TOP_K_VALUES
        ))
    save_json(args.output_report, {
        "experiment": "build7_automl_hidden_delta_routing_dynamics_topk_evaluation",
        "checkpoint": str(args.checkpoint),
        "architecture_report": str(args.architecture_report),
        "winner_trial_id": trial_id,
        "architecture": configuration,
        "selection_split": "validation", "test_split_used_for_selection": False,
        "metrics_definition": {
            "recall_at_k": "Frazione dei sei expert target presenti nei k score piu' alti.",
            "target_rank_distribution": "Posizione esatta dei target tra 256 score decrescenti.",
        },
        "splits_evaluated": splits,
        **results,
    })
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
