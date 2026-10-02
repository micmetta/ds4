#!/usr/bin/env python3
"""Inspect how the gated residual transition branch affects a trained FFN.

This is inference only.  It streams each split in batches and records the
learned residual scale alpha, gate statistics, residual magnitude, and how
often the residual changes the historical FFN Top-6 ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_build6_expert_transition_ablation import (
    load_keras_weights,
    load_split,
    make_dataset,
    parse_csv,
    select_candidate,
)
from ffn_utils import RANDOM_STATE, require_tensorflow, save_json
from ffn_utils import build_fixed_nnz_state_two_hidden_two_kv_ffn


TOP_K = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ispeziona gate e residuale q della FFN Build 6 gated residual."
    )
    parser.add_argument("--tuning-report", type=Path, required=True)
    parser.add_argument("--candidate-id", default="candidate_10")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--current-kv-data-dir", type=Path, required=True)
    parser.add_argument("--previous-kv-data-dir", type=Path, required=True)
    parser.add_argument("--token-position-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def top_k_indices(values: np.ndarray) -> np.ndarray:
    return np.argpartition(values, -TOP_K, axis=1)[:, -TOP_K:]


def recall_at_6_batch(values: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    indices = top_k_indices(values)
    rows = np.arange(values.shape[0])[:, None]
    return float(labels[rows, indices].sum()), int(values.shape[0])


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def build_model(args: argparse.Namespace, tf, reference: dict):
    candidate = select_candidate(args.tuning_report, args.candidate_id)
    hp = candidate["hyperparameters"]
    current_kv = None if hp["current_kv_projection"] == "direct" else int(hp["current_kv_projection"])
    previous_kv = None if hp["previous_kv_projection"] == "direct" else int(hp["previous_kv_projection"])
    transition = 0 if hp["expert_transition_projection"] == "direct" else int(hp["expert_transition_projection"])
    fusion = [] if hp["fusion_hidden_units"] == 0 else [hp["fusion_hidden_units"]]
    model = build_fixed_nnz_state_two_hidden_two_kv_ffn(
        reference["metadata"]["feature_count"], reference["metadata"]["input_slots"],
        hp["state_hidden_units"], fusion, "relu", 0.3, hp["learning_rate"], 0.0, tf,
        hp["current_raw_hidden_projection_dimension"], hp["previous_raw_hidden_projection_dimension"],
        reference["current_scale"], reference["previous_scale"], current_kv, previous_kv,
        ranking_loss_weight=0.0,
        ranking_loss_margin=0.05,
        expert_transition_projection_dimension=transition,
        expert_transition_gated_residual=True,
    )
    return model, hp


def inspect_split(probe, data: dict, batch_size: int, tf) -> dict[str, object]:
    """Accumulate diagnostics without materialising all 256D model outputs."""
    gate_values: list[np.ndarray] = []
    examples = 0
    final_hits = 0.0
    base_hits = 0.0
    q_only_hits = 0.0
    changed_top6 = 0
    overlap_sum = 0.0
    absolute_residual_sum = 0.0
    absolute_base_sum = 0.0
    residual_sum = 0.0
    residual_squared_sum = 0.0
    scalar_count = 0

    for inputs, labels_tensor in make_dataset(data, batch_size, tf):
        base_logits, final_logits, residual, gate = probe(inputs, training=False)
        base = base_logits.numpy()
        final = final_logits.numpy()
        residual_values = residual.numpy()
        gate_values.append(gate.numpy().reshape(-1))
        labels = labels_tensor.numpy()
        q_values = inputs["expert_transition_probabilities"].numpy()

        hits, count = recall_at_6_batch(final, labels)
        final_hits += hits
        hits, _ = recall_at_6_batch(base, labels)
        base_hits += hits
        hits, _ = recall_at_6_batch(q_values, labels)
        q_only_hits += hits
        examples += count

        base_top6 = top_k_indices(base)
        final_top6 = top_k_indices(final)
        overlaps = np.fromiter(
            (np.isin(base_row, final_row).sum() for base_row, final_row in zip(base_top6, final_top6)),
            dtype=np.int64,
            count=count,
        )
        overlap_sum += float(overlaps.sum())
        changed_top6 += int(np.count_nonzero(overlaps < TOP_K))
        absolute_residual_sum += float(np.abs(residual_values).sum())
        absolute_base_sum += float(np.abs(base).sum())
        residual_sum += float(residual_values.sum())
        residual_squared_sum += float(np.square(residual_values).sum())
        scalar_count += int(residual_values.size)

    gates = np.concatenate(gate_values)
    residual_mean = residual_sum / scalar_count
    residual_std = float(max(residual_squared_sum / scalar_count - residual_mean**2, 0.0) ** 0.5)
    return {
        "examples": examples,
        "recall_at_6": {
            "final_gated_model": final_hits / (examples * TOP_K),
            "base_ffn_logits_without_residual": base_hits / (examples * TOP_K),
            "transition_q_only": q_only_hits / (examples * TOP_K),
        },
        "gate_g_x": {
            **numeric_summary(gates),
            "fraction_below_0_1": float(np.mean(gates < 0.1)),
            "fraction_above_0_9": float(np.mean(gates > 0.9)),
        },
        "residual_vs_base_logits": {
            "mean_absolute_residual": absolute_residual_sum / scalar_count,
            "mean_absolute_base_logit": absolute_base_sum / scalar_count,
            "mean_absolute_residual_to_base_ratio": (
                absolute_residual_sum / absolute_base_sum if absolute_base_sum else None
            ),
            "residual_mean": residual_mean,
            "residual_standard_deviation": residual_std,
        },
        "top_6_effect": {
            "fraction_examples_with_changed_top_6": changed_top6 / examples,
            "mean_common_experts_between_base_and_final_top_6": overlap_sum / examples,
        },
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size deve essere positivo.")
    if not args.tuning_report.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("Tuning report o checkpoint .keras non trovato.")
    if args.output_report.exists():
        raise FileExistsError(f"Report gia' presente: {args.output_report}")
    splits = parse_csv(args.splits, ("train", "validation", "test"), "--splits")

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    print("1) Carico gli split e ricostruisco la FFN con residuale gated...")
    data_by_split = {split: load_split(args, split) for split in splits}
    model, hyperparameters = build_model(args, tf, data_by_split[splits[0]])
    load_keras_weights(args.checkpoint, model)

    alpha_weights = model.get_layer("expert_transition_residual_scale").get_weights()
    alpha = float(alpha_weights[0][0, 0])
    probe = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer("expert_base_logits").output,
            model.get_layer("expert_final_logits").output,
            model.get_layer("expert_transition_gated_scores").output,
            model.get_layer("expert_transition_gate").output,
        ],
    )
    print(f"2) Alpha appreso: {alpha:.6f}. Misuro gate, residuale e ranking...")
    results = {}
    for split in splits:
        result = inspect_split(probe, data_by_split[split], args.batch_size, tf)
        results[split] = result
        print(
            f"   {split}: Recall@6 finale={result['recall_at_6']['final_gated_model']:.6f}; "
            f"base={result['recall_at_6']['base_ffn_logits_without_residual']:.6f}; "
            f"gate medio={result['gate_g_x']['mean']:.4f}; "
            f"Top-6 modificata={result['top_6_effect']['fraction_examples_with_changed_top_6']:.2%}"
        )

    save_json(args.output_report, {
        "experiment": "build6_expert_transition_gated_residual_diagnostics",
        "checkpoint": str(args.checkpoint),
        "tuning_report": str(args.tuning_report),
        "candidate_id": args.candidate_id,
        "alpha_learned": alpha,
        "description": {
            "gate_g_x": "Distribuzione dello scalare sigmoid g(x) per esempio.",
            "residual_vs_base_logits": "Ampiezza del residuale q rispetto ai logit storici della FFN.",
            "top_6_effect": "Quanto il residuale modifica la Top-6 dei soli logit base.",
        },
        "hyperparameters": hyperparameters,
        "results": results,
    })
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
