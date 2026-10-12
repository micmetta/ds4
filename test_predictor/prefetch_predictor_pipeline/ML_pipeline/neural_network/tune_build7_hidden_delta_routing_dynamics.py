#!/usr/bin/env python3
"""Successive-halving search for the validated Build 7 candidate branches.

The search never reads the test split.  It starts a reproducible set of
architectures on a short budget, retains only the best validation Recall@6
configurations, and continues their best checkpoints on longer budgets.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from ffn_utils import (
    RANDOM_STATE,
    build_fixed_nnz_state_two_hidden_gated_residual_two_candidate_ffn,
    load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_quantized_hidden_inputs,
    require_new_output_directory,
    require_tensorflow,
    save_json,
)
from train_build7_routing_dynamics_hidden_delta import candidates, dataset


BASELINE_CONFIGURATION = {
    "state_hidden_units": 2048,
    "current_hidden_projection_dimension": 64,
    "previous_hidden_projection_dimension": 64,
    "fusion_hidden_units": 2048,
    "dropout": 0.30,
    "learning_rate": 1.5e-4,
    "l2_regularization": 3e-6,
    "hidden_delta_units": 64,
    "routing_dynamics_units": 128,
}


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("Attesa una lista non vuota di interi positivi.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-features-dir", type=Path, required=True)
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--current-fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=24,
                        help="Numero totale di configurazioni nella prima fase, inclusa la baseline.")
    parser.add_argument("--stage-epochs", type=parse_int_list, default=[10, 40, 150],
                        help="Budget cumulativi per fase, per esempio 10,40,150.")
    parser.add_argument("--survivors", type=parse_int_list, default=[8, 3],
                        help="Quanti trial conservare dopo ogni fase non finale, per esempio 8,3.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.trials < 2 or args.batch_size < 1:
        raise ValueError("--trials deve essere almeno 2 e --batch-size positivo.")
    if len(args.stage_epochs) < 2 or len(args.survivors) != len(args.stage_epochs) - 1:
        raise ValueError("--stage-epochs richiede almeno due valori e --survivors un valore per ogni fase non finale.")
    if any(next_epoch <= previous_epoch for previous_epoch, next_epoch in zip(args.stage_epochs, args.stage_epochs[1:])):
        raise ValueError("I budget in --stage-epochs devono essere strettamente crescenti.")
    if any(count < 1 for count in args.survivors) or args.survivors[0] >= args.trials:
        raise ValueError("Il primo valore di --survivors deve essere positivo e minore di --trials.")
    if any(next_count >= current_count for current_count, next_count in zip(args.survivors, args.survivors[1:])):
        raise ValueError("--survivors deve decrescere strettamente.")


def make_configurations(trials: int, seed: int) -> list[dict[str, float | int]]:
    """Return a fixed compact search space, always including the known baseline."""
    space = {
        "state_hidden_units": [1024, 2048, 3072],
        "current_hidden_projection_dimension": [48, 64, 96],
        "previous_hidden_projection_dimension": [48, 64, 96],
        "fusion_hidden_units": [1024, 2048, 3072],
        "dropout": [0.20, 0.30, 0.40],
        "learning_rate": [1.0e-4, 1.5e-4, 2.0e-4],
        "l2_regularization": [1.0e-6, 3.0e-6, 1.0e-5],
        "hidden_delta_units": [32, 64, 128],
        "routing_dynamics_units": [64, 128, 256],
    }
    keys = tuple(space)
    rng = np.random.default_rng(seed)
    configurations = [dict(BASELINE_CONFIGURATION)]
    seen = {tuple(configurations[0][key] for key in keys)}
    while len(configurations) < trials:
        candidate = {key: rng.choice(values).item() for key, values in space.items()}
        marker = tuple(candidate[key] for key in keys)
        if marker not in seen:
            seen.add(marker)
            configurations.append(candidate)
    return configurations


def save_ranking(path: Path, stage: int, trials: list[dict]) -> None:
    ranking = sorted(trials, key=lambda trial: trial["best_validation_recall_at_6"], reverse=True)
    save_json(path, {
        "stage": stage,
        "ranking": [
            {
                "trial_id": trial["trial_id"],
                "best_validation_recall_at_6": trial["best_validation_recall_at_6"],
                "trained_epochs": trial["trained_epochs"],
                "configuration": trial["configuration"],
            }
            for trial in ranking
        ],
    })


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_new_output_directory(args.output_dir)
    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")

    print("1) Carico Build 7 Top-4 e le due feature validate una sola volta...")
    y_train = load_labels(args.state_data_dir, "train")
    y_validation = load_labels(args.state_data_dir, "validation")
    train_indices, train_values, state_meta = load_fixed_nnz_state_inputs(
        args.current_fixed_nnz_state_data_dir, "train", y_train.shape[0]
    )
    validation_indices, validation_values, validation_state_meta = load_fixed_nnz_state_inputs(
        args.current_fixed_nnz_state_data_dir, "validation", y_validation.shape[0]
    )
    if state_meta != validation_state_meta:
        raise ValueError("Metadata state train/validation diverse.")
    current_train, current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, "train", y_train.shape[0]
    )
    current_validation, validation_current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, "validation", y_validation.shape[0]
    )
    previous_train, previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, "train", y_train.shape[0]
    )
    previous_validation, validation_previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, "validation", y_validation.shape[0]
    )
    if (current_scale, previous_scale) != (validation_current_scale, validation_previous_scale):
        raise ValueError("Scale raw hidden train/validation diverse.")
    transition_train = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, "train", y_train.shape[0]
    )
    transition_validation = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, "validation", y_validation.shape[0]
    )
    delta_train, delta_aux_train, routing_train, delta_scale = candidates(
        args.candidate_features_dir, "train", y_train.shape[0]
    )
    delta_validation, delta_aux_validation, routing_validation, validation_delta_scale = candidates(
        args.candidate_features_dir, "validation", y_validation.shape[0]
    )
    if delta_scale != validation_delta_scale:
        raise ValueError("Scale hidden delta train/validation diversa.")
    print(f"   train={y_train.shape[0]:,}; validation={y_validation.shape[0]:,}; "
          f"delta={delta_train.shape}; routing={routing_train.shape}")

    train_steps = math.ceil(y_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / args.batch_size)
    configurations = make_configurations(args.trials, args.seed)
    trials = [
        {
            "trial_id": f"trial_{index:03d}",
            "configuration": configuration,
            "trained_epochs": 0,
            "best_validation_recall_at_6": -1.0,
            "history": [],
        }
        for index, configuration in enumerate(configurations)
    ]
    models_dir = args.output_dir / "models"
    reports_dir = args.output_dir / "reports"
    models_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    save_json(reports_dir / "search_configuration.json", {
        "experiment": "build7_top4_hidden_delta_routing_dynamics_successive_halving",
        "trials": args.trials,
        "stage_epochs": args.stage_epochs,
        "survivors": args.survivors,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "baseline_configuration": BASELINE_CONFIGURATION,
        "all_configurations": trials,
    })

    active = trials
    for stage, target_epochs in enumerate(args.stage_epochs):
        print(f"\n2.{stage + 1}) Fase {stage + 1}/{len(args.stage_epochs)}: "
              f"{len(active)} trial fino a {target_epochs} epoche...")
        for position, trial in enumerate(active, start=1):
            configuration = trial["configuration"]
            trial_id = trial["trial_id"]
            checkpoint = models_dir / f"{trial_id}.best.weights.h5"
            tf.keras.backend.clear_session()
            gc.collect()
            tf.keras.utils.set_random_seed(args.seed + int(trial_id.rsplit("_", 1)[1]))
            model = build_fixed_nnz_state_two_hidden_gated_residual_two_candidate_ffn(
                state_meta["feature_count"], state_meta["input_slots"],
                int(configuration["state_hidden_units"]),
                int(configuration["current_hidden_projection_dimension"]),
                int(configuration["previous_hidden_projection_dimension"]),
                current_scale, previous_scale, int(configuration["fusion_hidden_units"]),
                float(configuration["dropout"]), float(configuration["learning_rate"]),
                float(configuration["l2_regularization"]),
                4096, int(configuration["hidden_delta_units"]), "int8", delta_scale, 4,
                261, int(configuration["routing_dynamics_units"]), "float16", tf,
            )
            previous_best = float(trial["best_validation_recall_at_6"])
            if checkpoint.is_file():
                model.load_weights(checkpoint)
            train_data = dataset(
                train_indices, train_values, y_train, current_train, previous_train,
                transition_train, delta_train, delta_aux_train, routing_train,
                args.batch_size, True, tf,
            ).repeat()
            validation_data = dataset(
                validation_indices, validation_values, y_validation,
                current_validation, previous_validation, transition_validation,
                delta_validation, delta_aux_validation, routing_validation,
                args.batch_size, False, tf,
            ).repeat()
            callback_kwargs = {}
            if previous_best >= 0:
                callback_kwargs["initial_value_threshold"] = previous_best
            callbacks = [tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint), monitor="val_recall_at_6", mode="max", save_best_only=True,
                save_weights_only=True, **callback_kwargs,
            )]
            print(f"   [{position}/{len(active)}] {trial_id}: "
                  f"epoche {trial['trained_epochs'] + 1}--{target_epochs}; {configuration}")
            started = time.perf_counter()
            history = model.fit(
                train_data, steps_per_epoch=train_steps,
                validation_data=validation_data, validation_steps=validation_steps,
                initial_epoch=int(trial["trained_epochs"]), epochs=target_epochs,
                callbacks=callbacks, verbose=2,
            )
            elapsed = time.perf_counter() - started
            stage_best = max(float(value) for value in history.history["val_recall_at_6"])
            trial["best_validation_recall_at_6"] = max(previous_best, stage_best)
            trial["trained_epochs"] = target_epochs
            trial["history"].append({
                "stage": stage + 1,
                "target_epochs": target_epochs,
                "elapsed_seconds": elapsed,
                "best_validation_recall_at_6_in_stage": stage_best,
                "history": {key: [float(value) for value in values] for key, values in history.history.items()},
            })
            save_json(reports_dir / f"{trial_id}.json", trial)
            del model, train_data, validation_data
            gc.collect()
        save_ranking(reports_dir / f"stage_{stage + 1}_ranking.json", stage + 1, active)
        if stage < len(args.survivors):
            keep = args.survivors[stage]
            active = sorted(active, key=lambda trial: trial["best_validation_recall_at_6"], reverse=True)[:keep]
            print("   Promossi:", ", ".join(
                f"{trial['trial_id']}={trial['best_validation_recall_at_6']:.6f}" for trial in active
            ))

    ranking = sorted(active, key=lambda trial: trial["best_validation_recall_at_6"], reverse=True)
    winner = ranking[0]
    source = models_dir / f"{winner['trial_id']}.best.weights.h5"
    destination = models_dir / "best.weights.h5"
    shutil.copy2(source, destination)
    save_json(reports_dir / "final_ranking.json", {
        "winner": winner,
        "finalists": ranking,
        "winner_weights_path": str(destination),
        "selection_split": "validation",
        "test_split_used": False,
    })
    print(f"\nCompletato. Vincitore {winner['trial_id']}: validation Recall@6 "
          f"{winner['best_validation_recall_at_6']:.6f}")


if __name__ == "__main__":
    main()
