#!/usr/bin/env python3
"""Train Build 7 with the fixed best Build 6 gated-residual architecture.

The architecture is deliberately fixed to the configuration that made the
transition-probability branch demonstrably relevant in Build 6.  Build 7
changes the fixed-NNZ state only by adding the causally available l-2 block.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

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
    make_fixed_nnz_state_two_hidden_two_kv_dataset,
    require_new_output_directory,
    require_tensorflow,
    save_json,
)


# Fixed configuration of the Build 6 candidate_10 gated-residual experiment.
STATE_HIDDEN_UNITS = 2048
CURRENT_HIDDEN_UNITS = 64
PREVIOUS_HIDDEN_UNITS = 64
CURRENT_KV_UNITS = 256
FUSION_HIDDEN_UNITS = 2048
LEARNING_RATE = 0.00015
DROPOUT = 0.3
L2_REGULARIZATION = 0.0
BRANCH_DROPOUT = 0.0
TRANSITION_LOG_EPSILON = 1e-6
TRANSITION_RESIDUAL_INITIAL_SCALE = 0.1


def load_positions(directory: Path, split: str, examples: int) -> np.ndarray:
    path = directory / "arrays" / f"token_position_{split}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"token_position non trovata: {path}")
    values = np.load(path, mmap_mode="r")
    if values.shape != (examples, 1) or values.dtype != np.float32:
        raise ValueError(f"token_position incompatibile: {path}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Addestra Build 7 con BCE e residuale gated delle probabilita' di transizione."
    )
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--current-kv-data-dir", type=Path, required=True)
    parser.add_argument("--previous-kv-data-dir", type=Path, required=True)
    parser.add_argument("--token-position-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--shuffle-block-size", type=int, default=65536)
    parser.add_argument(
        "--l2-regularization", type=float, default=L2_REGULARIZATION,
        help="Penalizzazione L2 dei pesi; 0 riproduce il baseline senza weight decay.",
    )
    parser.add_argument(
        "--state-hidden-units", type=int, default=STATE_HIDDEN_UNITS,
        help="Ampiezza del ramo state fixed-NNZ; 2048 riproduce la configurazione Build 6.",
    )
    parser.add_argument(
        "--branch-dropout", type=float, default=BRANCH_DROPOUT,
        help="Dropout dopo ciascuna proiezione raw-hidden/KV; 0 riproduce il baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.epochs < 1 or args.early_stopping_patience < 0 or args.batch_size < 1
            or args.state_hidden_units < 1
            or args.l2_regularization < 0
            or args.branch_dropout < 0 or args.branch_dropout >= 1
            or args.shuffle_block_size < args.batch_size):
        raise ValueError("Parametri di training non validi.")
    require_new_output_directory(args.output_dir)

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("1) Carico Build 7: state l/l-1/l-2, hidden/KV l/l-1, token_position e q 256D...")
    y_train = load_labels(args.state_data_dir, "train")
    y_validation = load_labels(args.state_data_dir, "validation")
    train_indices, train_values, metadata = load_fixed_nnz_state_inputs(
        args.fixed_nnz_state_data_dir, "train", y_train.shape[0]
    )
    validation_indices, validation_values, validation_metadata = load_fixed_nnz_state_inputs(
        args.fixed_nnz_state_data_dir, "validation", y_validation.shape[0]
    )
    if metadata != validation_metadata:
        raise ValueError("Metadata state train e validation diverse.")
    train_position = load_positions(args.token_position_data_dir, "train", y_train.shape[0])
    validation_position = load_positions(args.token_position_data_dir, "validation", y_validation.shape[0])
    train_current_hidden, current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, "train", y_train.shape[0]
    )
    validation_current_hidden, validation_current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, "validation", y_validation.shape[0]
    )
    train_previous_hidden, previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, "train", y_train.shape[0]
    )
    validation_previous_hidden, validation_previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, "validation", y_validation.shape[0]
    )
    if current_scale != validation_current_scale or previous_scale != validation_previous_scale:
        raise ValueError("Scale hidden train e validation diverse.")
    train_current_kv = load_materialized_kv_inputs(args.current_kv_data_dir, "train", y_train.shape[0])
    validation_current_kv = load_materialized_kv_inputs(args.current_kv_data_dir, "validation", y_validation.shape[0])
    train_previous_kv = load_materialized_kv_inputs(args.previous_kv_data_dir, "train", y_train.shape[0])
    validation_previous_kv = load_materialized_kv_inputs(args.previous_kv_data_dir, "validation", y_validation.shape[0])
    train_transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, "train", y_train.shape[0]
    )
    validation_transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, "validation", y_validation.shape[0]
    )
    print(f"   train: state={train_indices.shape}; q={train_transition.shape}; "
          f"hidden={train_current_hidden.shape}/{train_previous_hidden.shape}; "
          f"KV={train_current_kv.shape}/{train_previous_kv.shape}")

    print("2) Creo la configurazione Build 6 candidate_10 con residuale gated q, da zero...")
    print(f"   Ramo state Build 7: {args.state_hidden_units} unita'; "
          f"L2={args.l2_regularization:g}; branch dropout={args.branch_dropout:g}.")
    model = build_fixed_nnz_state_two_hidden_two_kv_ffn(
        metadata["feature_count"], metadata["input_slots"], args.state_hidden_units,
        [FUSION_HIDDEN_UNITS], "relu", DROPOUT, LEARNING_RATE, args.l2_regularization, tf,
        CURRENT_HIDDEN_UNITS, PREVIOUS_HIDDEN_UNITS, current_scale, previous_scale,
        CURRENT_KV_UNITS, None,
        ranking_loss_weight=0.0,
        ranking_loss_margin=0.05,
        expert_transition_projection_dimension=0,
        expert_transition_gated_residual=True,
        transition_log_epsilon=TRANSITION_LOG_EPSILON,
        transition_residual_initial_scale=TRANSITION_RESIDUAL_INITIAL_SCALE,
        branch_dropout=args.branch_dropout,
    )
    model.summary()

    train_dataset = make_fixed_nnz_state_two_hidden_two_kv_dataset(
        train_indices, train_values, y_train, train_position, train_current_hidden,
        train_previous_hidden, train_current_kv, train_previous_kv, args.batch_size,
        True, tf, args.shuffle_block_size, train_transition,
    ).repeat()
    validation_dataset = make_fixed_nnz_state_two_hidden_two_kv_dataset(
        validation_indices, validation_values, y_validation, validation_position,
        validation_current_hidden, validation_previous_hidden, validation_current_kv,
        validation_previous_kv, args.batch_size, False, tf, args.shuffle_block_size,
        validation_transition,
    )
    train_steps = math.ceil(y_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / args.batch_size)
    models_dir, reports_dir = args.output_dir / "models", args.output_dir / "reports"
    models_dir.mkdir(parents=True)
    best_path = models_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path), monitor="val_recall_at_6", mode="max",
            save_best_only=True, save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall_at_6", mode="max", patience=args.early_stopping_patience,
            restore_best_weights=True,
        ),
    ]
    print(f"3) Addestro BCE fino a {args.epochs} epoche con patience={args.early_stopping_patience}...")
    started = time.perf_counter()
    history = model.fit(
        train_dataset, steps_per_epoch=train_steps,
        validation_data=validation_dataset.repeat(), validation_steps=validation_steps,
        epochs=args.epochs, callbacks=callbacks, shuffle=False, verbose=2,
    )
    elapsed = time.perf_counter() - started
    model.load_weights(best_path)
    scores = model.predict(validation_dataset.repeat(), steps=validation_steps, verbose=0)
    recall = calculate_recall_at_6(scores, y_validation)
    model_path = models_dir / "ffn.keras"
    model.save(model_path)
    save_json(reports_dir / "training_report.json", {
        "experiment": "build7_two_previous_layers_expert_transition_gated_residual_bce",
        "best_validation_recall_at_6": recall,
        "elapsed_seconds": elapsed,
        "loss": "binary_cross_entropy",
        "state_fixed_nnz_metadata": metadata,
        "architecture": {
            "source": "build6_candidate_10_gated_residual",
            "state_hidden_units": args.state_hidden_units,
            "current_hidden_projection_dimension": CURRENT_HIDDEN_UNITS,
            "previous_hidden_projection_dimension": PREVIOUS_HIDDEN_UNITS,
            "current_kv_projection_dimension": CURRENT_KV_UNITS,
            "previous_kv_projection_dimension": CURRENT_KV_UNITS,
            "fusion_hidden_layers": [FUSION_HIDDEN_UNITS],
            "expert_transition_projection": "direct_gated_residual",
            "transition_log_epsilon": TRANSITION_LOG_EPSILON,
            "transition_residual_initial_scale": TRANSITION_RESIDUAL_INITIAL_SCALE,
            "branch_dropout": args.branch_dropout,
        },
        "hyperparameters": {
            "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
            "l2_regularization": args.l2_regularization, "batch_size": args.batch_size,
            "branch_dropout": args.branch_dropout,
            "shuffle_block_size": args.shuffle_block_size, "epochs_max": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "model_path": str(model_path), "best_weights_path": str(best_path),
    })
    print(f"Build 7 gated-residual training completato: validation Recall@6 {recall:.6f}")
    print(f"Modello: {model_path}")


if __name__ == "__main__":
    main()
