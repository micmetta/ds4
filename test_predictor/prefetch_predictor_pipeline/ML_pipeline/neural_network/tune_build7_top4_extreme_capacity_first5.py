#!/usr/bin/env python3
"""Build 7 Top-4 tuning C: eight extreme-capacity models, first five epochs.

Every completed epoch is persisted atomically with the latest model and Adam
state.  An interruption can therefore lose at most the currently running,
incomplete epoch; rerun with --resume to continue safely.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import tune_build7_top4_successive_halving as shared
from ffn_utils import (
    RANDOM_STATE,
    build_fixed_nnz_state_two_hidden_gated_residual_ffn,
    load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_quantized_hidden_inputs,
    require_tensorflow,
)


EPOCHS_PER_CANDIDATE = 5
STATE_UNITS = (16384, 32768)
HIDDEN_UNITS = (8192, 16384)
FUSION_UNITS = (8192, 16384)
EXPECTED_STATE_METADATA = {
    "feature_count": 4411,
    "nonzero_features_per_example": 44,
    "input_slots": 45,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuning C Build 7 Top-4: 8 configurazioni estreme, esattamente 5 epoche ciascuna."
    )
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--current-fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Riprende dall'ultima epoca completa salvata.")
    parser.add_argument("--l2-regularization", type=float, default=3e-6)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--shuffle-block-size", type=int, default=65536)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (args.batch_size < 1 or args.shuffle_block_size < args.batch_size
            or args.l2_regularization < 0 or args.learning_rate <= 0
            or not 0 <= args.dropout < 1):
        raise ValueError("Iperparametri del tuning C non validi.")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output gia' esistente: {args.output_dir}. Usa --resume o una nuova cartella.")
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError(f"Nessun tuning C da riprendere in: {args.output_dir}")


def all_candidates() -> list[dict[str, Any]]:
    candidates = []
    number = 1
    for state_units in STATE_UNITS:
        for hidden_units in HIDDEN_UNITS:
            for fusion_units in FUSION_UNITS:
                candidates.append({
                    "candidate_id": f"candidate_C{number:02d}",
                    "hyperparameters": {
                        "state_hidden_units": state_units,
                        "current_hidden_projection_dimension": hidden_units,
                        "previous_hidden_projection_dimension": hidden_units,
                        "fusion_hidden_units": fusion_units,
                        "raw_hidden_branch_dropout": 0.0,
                    },
                    "status": "pending",
                    "completed_epochs": 0,
                    "best_validation_recall_at_6": None,
                    "history": {},
                })
                number += 1
    if len(candidates) != 8:
        raise AssertionError("Il tuning C deve contenere esattamente 8 candidate.")
    return candidates


def immutable_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "build7_top4_extreme_capacity_first5",
        "purpose": "Misurare soltanto le prime cinque epoche di otto configurazioni estreme.",
        "data": {
            "state_data_dir": str(args.state_data_dir),
            "current_fixed_nnz_state_data_dir": str(args.current_fixed_nnz_state_data_dir),
            "current_quantized_hidden_data_dir": str(args.current_quantized_hidden_data_dir),
            "previous_quantized_hidden_data_dir": str(args.previous_quantized_hidden_data_dir),
            "expert_transition_probabilities_data_dir": str(args.expert_transition_probabilities_data_dir),
        },
        "search_space": {
            "state_hidden_units": list(STATE_UNITS),
            "symmetric_raw_hidden_projection_dimension": list(HIDDEN_UNITS),
            "fusion_hidden_units": list(FUSION_UNITS),
            "candidate_count": 8,
            "epochs_per_candidate": EPOCHS_PER_CANDIDATE,
        },
        "fixed_hyperparameters": {
            "loss": "binary_cross_entropy",
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
            "state_and_fusion_dropout": args.dropout,
            "raw_hidden_branch_dropout": 0.0,
            "batch_size": args.batch_size,
            "shuffle_block_size": args.shuffle_block_size,
            "tensorflow_policy": "float32",
        },
        "selection_metric": "validation Recall@6 at each of the first five epochs",
        "test_set_used": False,
        "random_state": RANDOM_STATE,
    }


def checkpoint_path(root: Path, candidate_id: str, kind: str) -> Path:
    return root / "candidate_checkpoints" / candidate_id / f"{kind}.model_and_adam.npz"


def main() -> None:
    args = parse_args()
    validate_args(args)
    reports_dir = args.output_dir / "reports"
    configuration = immutable_configuration(args)
    configuration_path = reports_dir / "search_configuration.json"
    state_path = reports_dir / "tuning_state.json"
    if args.resume:
        if not configuration_path.is_file() or not state_path.is_file():
            raise FileNotFoundError("Mancano configurazione o stato: impossibile riprendere in sicurezza.")
        if json.loads(configuration_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("La configurazione richiesta differisce dal tuning C gia' iniziato.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        args.output_dir.mkdir(parents=True)
        state = {
            "experiment": configuration["experiment"],
            "candidates": all_candidates(),
            "interrupted": False,
        }
        shared.write_json_atomically(configuration_path, configuration)
        shared.write_json_atomically(state_path, state)

    def save_state() -> None:
        shared.write_json_atomically(state_path, state)

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    print("1) Carico Build 7 Top-4 per Tuning C: state l, raw hidden l/l-1 e q 256D...")
    y_train = load_labels(args.state_data_dir, "train")
    y_validation = load_labels(args.state_data_dir, "validation")
    train_indices, train_values, metadata = load_fixed_nnz_state_inputs(
        args.current_fixed_nnz_state_data_dir, "train", y_train.shape[0]
    )
    validation_indices, validation_values, validation_metadata = load_fixed_nnz_state_inputs(
        args.current_fixed_nnz_state_data_dir, "validation", y_validation.shape[0]
    )
    if metadata != EXPECTED_STATE_METADATA or validation_metadata != metadata:
        raise ValueError("Il tuning C richiede lo state corrente 4411D con 45 slot.")
    train_current, current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, "train", y_train.shape[0]
    )
    validation_current, validation_current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, "validation", y_validation.shape[0]
    )
    train_previous, previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, "train", y_train.shape[0]
    )
    validation_previous, validation_previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, "validation", y_validation.shape[0]
    )
    if current_scale != validation_current_scale or previous_scale != validation_previous_scale:
        raise ValueError("Scale int8 diverse tra train e validation.")
    train_transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, "train", y_train.shape[0]
    )
    validation_transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, "validation", y_validation.shape[0]
    )
    train_steps = math.ceil(y_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / args.batch_size)
    print(f"   train: state={train_indices.shape}; hidden={train_current.shape}/{train_previous.shape}; q={train_transition.shape}")

    invocation_started = time.perf_counter()
    interrupted = False
    try:
        print(f"2) Tuning C: {len(state['candidates'])} candidate, esattamente {EPOCHS_PER_CANDIDATE} epoche ciascuna.")
        for position, candidate in enumerate(state["candidates"], start=1):
            completed_epochs = int(candidate["completed_epochs"])
            if completed_epochs >= EPOCHS_PER_CANDIDATE:
                continue
            hp = candidate["hyperparameters"]
            candidate_id = candidate["candidate_id"]
            print(f"   {position}/{len(state['candidates'])} {candidate_id}: state={hp['state_hidden_units']}, "
                  f"h={hp['current_hidden_projection_dimension']}/{hp['previous_hidden_projection_dimension']}, "
                  f"fusion={hp['fusion_hidden_units']}; riprendo da epoca {completed_epochs}.")
            tf.keras.backend.clear_session()
            gc.collect()
            tf.keras.utils.set_random_seed(RANDOM_STATE)
            model = build_fixed_nnz_state_two_hidden_gated_residual_ffn(
                metadata["feature_count"], metadata["input_slots"], hp["state_hidden_units"],
                hp["current_hidden_projection_dimension"], hp["previous_hidden_projection_dimension"],
                current_scale, previous_scale, hp["fusion_hidden_units"], args.dropout,
                args.learning_rate, args.l2_regularization, tf,
            )
            latest_checkpoint = checkpoint_path(args.output_dir, candidate_id, "latest")
            best_checkpoint = checkpoint_path(args.output_dir, candidate_id, "best_validation")
            if completed_epochs:
                if not latest_checkpoint.is_file():
                    raise FileNotFoundError(f"Manca il checkpoint latest per {candidate_id}: {latest_checkpoint}")
                shared.load_model_and_adam(latest_checkpoint, model)
            candidate["status"] = "running"
            save_state()
            try:
                for epoch in range(completed_epochs, EPOCHS_PER_CANDIDATE):
                    train_dataset = shared.make_dataset(
                        train_indices, train_values, y_train, train_current, train_previous,
                        train_transition, args.batch_size, True, epoch, tf,
                    ).repeat()
                    validation_dataset = shared.make_dataset(
                        validation_indices, validation_values, y_validation, validation_current,
                        validation_previous, validation_transition, args.batch_size, False, 0, tf,
                    ).repeat()
                    history = model.fit(
                        train_dataset, steps_per_epoch=train_steps, validation_data=validation_dataset,
                        validation_steps=validation_steps, initial_epoch=epoch, epochs=epoch + 1,
                        shuffle=False, verbose=2,
                    )
                    for metric, values in history.history.items():
                        candidate["history"].setdefault(metric, []).extend(float(value) for value in values)
                    score = float(history.history["val_recall_at_6"][-1])
                    shared.save_model_and_adam(latest_checkpoint, model)
                    previous_best = candidate["best_validation_recall_at_6"]
                    if previous_best is None or score > float(previous_best):
                        shared.save_model_and_adam(best_checkpoint, model)
                        candidate["best_validation_recall_at_6"] = score
                        candidate["best_validation_epoch"] = epoch + 1
                    candidate["completed_epochs"] = epoch + 1
                    candidate["last_completed_epoch_elapsed_seconds"] = time.perf_counter() - invocation_started
                    save_state()
            finally:
                del model
                tf.keras.backend.clear_session()
                gc.collect()
            candidate["status"] = "completed"
            save_state()
    except KeyboardInterrupt:
        interrupted = True
        state["interrupted"] = True
        save_state()
        shared.write_json_atomically(reports_dir / "interruption_report.json", {
            "message": "Tuning C interrotto: riesegui lo stesso comando con --resume.",
            "completed_epochs": {
                candidate["candidate_id"]: candidate["completed_epochs"] for candidate in state["candidates"]
            },
        })
        print("\nTuning C interrotto. Riprendi con lo stesso comando e --resume.")

    ranking = sorted(
        (candidate for candidate in state["candidates"] if candidate["best_validation_recall_at_6"] is not None),
        key=lambda candidate: float(candidate["best_validation_recall_at_6"]), reverse=True,
    )
    shared.write_json_atomically(reports_dir / "tuning_report.json", {
        "experiment": configuration["experiment"],
        "interrupted": interrupted,
        "elapsed_seconds_this_invocation": time.perf_counter() - invocation_started,
        "epochs_per_candidate": EPOCHS_PER_CANDIDATE,
        "ranking_by_best_validation_recall_at_6": ranking,
        "next_step": {
            "automatic": False,
            "message": "Questo e' un esperimento limitato alle prime cinque epoche; non avviare training prolungati automaticamente.",
        },
    })
    if not interrupted and all(candidate["completed_epochs"] == EPOCHS_PER_CANDIDATE for candidate in state["candidates"]):
        print("\nTuning C completato: 8 candidate salvate dopo esattamente 5 epoche ciascuna.")
        print(f"Report: {reports_dir / 'tuning_report.json'}")


if __name__ == "__main__":
    main()
