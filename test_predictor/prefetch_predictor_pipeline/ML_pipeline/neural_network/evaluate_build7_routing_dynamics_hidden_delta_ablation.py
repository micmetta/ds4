#!/usr/bin/env python3
"""Measure independent hidden-delta and routing-dynamics contributions in the combined Build 7 model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ffn_utils import (
    RANDOM_STATE, build_fixed_nnz_state_two_hidden_gated_residual_two_candidate_ffn,
    calculate_recall_at_6, load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs, load_labels, load_quantized_hidden_inputs,
    require_tensorflow, save_json,
)
from train_build7_routing_dynamics_hidden_delta import candidates, dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-features-dir", type=Path, required=True)
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--current-fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file() or args.batch_size < 1:
        raise ValueError("Checkpoint o batch size non valido.")
    if args.output_report.exists():
        raise FileExistsError(f"Report gia' esistente: {args.output_report}")
    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    labels = load_labels(args.state_data_dir, "validation")
    indices, values, metadata = load_fixed_nnz_state_inputs(args.current_fixed_nnz_state_data_dir, "validation", labels.shape[0])
    current, current_scale = load_quantized_hidden_inputs(args.current_quantized_hidden_data_dir, "validation", labels.shape[0])
    previous, previous_scale = load_quantized_hidden_inputs(args.previous_quantized_hidden_data_dir, "validation", labels.shape[0])
    transition = load_expert_transition_probabilities_inputs(args.expert_transition_probabilities_data_dir, "validation", labels.shape[0])
    delta, delta_aux, routing, delta_scale = candidates(args.candidate_features_dir, "validation", labels.shape[0])
    model = build_fixed_nnz_state_two_hidden_gated_residual_two_candidate_ffn(
        metadata["feature_count"], metadata["input_slots"], 2048, 64, 64,
        current_scale, previous_scale, 2048, 0.3, 1.5e-4, 3e-6,
        4096, 64, "int8", delta_scale, 4, 261, 128, "float16", tf,
    )
    model.load_weights(args.checkpoint)
    steps = math.ceil(labels.shape[0] / args.batch_size)
    rng = np.random.default_rng(args.seed)
    delta_permutation = rng.permutation(len(delta))
    routing_permutation = rng.permutation(len(routing))
    modes = {
        "reference": (delta, delta_aux, routing),
        "hidden_delta_zero": (np.zeros_like(delta), np.zeros_like(delta_aux), routing),
        "hidden_delta_permutation": (delta[delta_permutation], delta_aux[delta_permutation], routing),
        "routing_dynamics_zero": (delta, delta_aux, np.zeros_like(routing)),
        "routing_dynamics_permutation": (delta, delta_aux, routing[routing_permutation]),
        "both_zero": (np.zeros_like(delta), np.zeros_like(delta_aux), np.zeros_like(routing)),
    }
    results = {}
    for name, (mode_delta, mode_aux, mode_routing) in modes.items():
        data = dataset(indices, values, labels, current, previous, transition, mode_delta, mode_aux, mode_routing,
                       args.batch_size, False, tf).repeat()
        results[name] = calculate_recall_at_6(model.predict(data, steps=steps, verbose=0), labels)
    reference = results["reference"]
    payload = {
        "experiment": "build7_top4_hidden_delta_routing_dynamics_ablation",
        "split": "validation", "checkpoint": str(args.checkpoint), "seed": args.seed,
        "recall_at_6": results,
        "delta_vs_reference": {name: value - reference for name, value in results.items() if name != "reference"},
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output_report, payload)
    print("validation: " + "; ".join(f"{name}={value:.6f}" for name, value in results.items()))
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
