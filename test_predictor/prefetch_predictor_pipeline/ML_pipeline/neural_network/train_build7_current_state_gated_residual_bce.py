#!/usr/bin/env python3
"""Train the deliberately minimal Build 7 model: current state plus q."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from ffn_utils import (
    RANDOM_STATE,
    build_fixed_nnz_state_gated_residual_ffn,
    calculate_recall_at_6,
    load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs,
    load_labels,
    require_new_output_directory,
    require_tensorflow,
    save_json,
)


STATE_HIDDEN_UNITS = 2048
FUSION_HIDDEN_UNITS = 2048
DROPOUT = 0.3
LEARNING_RATE = 0.00015
L2_REGULARIZATION = 3e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Addestra Build 7 minimale: state corrente l e transition probabilities gated residual."
    )
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--current-fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--shuffle-block-size", type=int, default=65536)
    parser.add_argument("--state-hidden-units", type=int, default=STATE_HIDDEN_UNITS)
    parser.add_argument("--fusion-hidden-units", type=int, default=FUSION_HIDDEN_UNITS)
    parser.add_argument("--l2-regularization", type=float, default=L2_REGULARIZATION)
    return parser.parse_args()


def make_dataset(indices, values, labels, transition, batch_size, training, tf, shuffle_block_size):
    examples, slots = indices.shape
    epoch_number = 0

    def batches():
        nonlocal epoch_number
        order = np.arange(examples)
        if training:
            rng = np.random.default_rng(RANDOM_STATE + epoch_number)
            rng.shuffle(order)
            epoch_number += 1
        for start in range(0, examples, batch_size):
            rows = order[start:min(start + batch_size, examples)] if training else slice(start, min(start + batch_size, examples))
            yield {
                "state_feature_indices": indices[rows].astype(np.int32, copy=False),
                "state_feature_values": values[rows].astype(np.float32, copy=False),
                "expert_transition_probabilities": transition[rows].astype(np.float16, copy=False),
            }, labels[rows].astype(np.float32, copy=False)

    signature = ({
        "state_feature_indices": tf.TensorSpec((None, slots), tf.int32),
        "state_feature_values": tf.TensorSpec((None, slots), tf.float32),
        "expert_transition_probabilities": tf.TensorSpec((None, 256), tf.float16),
    }, tf.TensorSpec((None, 256), tf.float32))
    # The generator is recreated for each repeat. Randomisation changes only
    # between epochs; no label or validation data is used in the permutation.
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    if (args.epochs < 1 or args.early_stopping_patience < 0 or args.batch_size < 1
            or args.shuffle_block_size < args.batch_size or args.state_hidden_units < 1
            or args.fusion_hidden_units < 1 or args.l2_regularization < 0):
        raise ValueError("Parametri di training non validi.")
    require_new_output_directory(args.output_dir)
    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("1) Carico Build 7 minimale: state corrente l (45 slot) e q 256D...")
    y_train = load_labels(args.state_data_dir, "train")
    y_validation = load_labels(args.state_data_dir, "validation")
    train_indices, train_values, metadata = load_fixed_nnz_state_inputs(
        args.current_fixed_nnz_state_data_dir, "train", y_train.shape[0]
    )
    validation_indices, validation_values, validation_metadata = load_fixed_nnz_state_inputs(
        args.current_fixed_nnz_state_data_dir, "validation", y_validation.shape[0]
    )
    if metadata != validation_metadata:
        raise ValueError("Metadata state train e validation diverse.")
    if metadata != {"feature_count": 4411, "nonzero_features_per_example": 44, "input_slots": 45}:
        raise ValueError("Il trainer minimale richiede lo state Build 7 corrente con 4411 feature e 45 slot.")
    train_transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, "train", y_train.shape[0]
    )
    validation_transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, "validation", y_validation.shape[0]
    )
    print(f"   train: state={train_indices.shape}; q={train_transition.shape}")

    print("2) Creo FFN minimale: state l -> fusione -> logits, con q gated residual...")
    print(f"   state={args.state_hidden_units}; fusione={args.fusion_hidden_units}; L2={args.l2_regularization:g}.")
    model = build_fixed_nnz_state_gated_residual_ffn(
        metadata["feature_count"], metadata["input_slots"], args.state_hidden_units,
        args.fusion_hidden_units, DROPOUT, LEARNING_RATE, args.l2_regularization, tf,
    )
    model.summary()
    train_dataset = make_dataset(
        train_indices, train_values, y_train, train_transition, args.batch_size,
        True, tf, args.shuffle_block_size,
    ).repeat()
    validation_dataset = make_dataset(
        validation_indices, validation_values, y_validation, validation_transition,
        args.batch_size, False, tf, args.shuffle_block_size,
    ).repeat()
    train_steps = math.ceil(y_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / args.batch_size)
    models_dir, reports_dir = args.output_dir / "models", args.output_dir / "reports"
    models_dir.mkdir(parents=True)
    best_path = models_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path), monitor="val_recall_at_6", mode="max", save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall_at_6", mode="max", patience=args.early_stopping_patience,
            restore_best_weights=True,
        ),
    ]
    print(f"3) Addestro BCE fino a {args.epochs} epoche con patience={args.early_stopping_patience}...")
    started = time.perf_counter()
    history = model.fit(
        train_dataset, steps_per_epoch=train_steps, validation_data=validation_dataset,
        validation_steps=validation_steps, epochs=args.epochs, callbacks=callbacks,
        shuffle=False, verbose=2,
    )
    elapsed = time.perf_counter() - started
    model.load_weights(best_path)
    scores = model.predict(validation_dataset, steps=validation_steps, verbose=0)
    recall = calculate_recall_at_6(scores, y_validation)
    model_path = models_dir / "ffn.keras"
    model.save(model_path)
    save_json(reports_dir / "training_report.json", {
        "experiment": "build7_current_state_only_expert_transition_gated_residual_bce",
        "best_validation_recall_at_6": recall,
        "elapsed_seconds": elapsed,
        "loss": "binary_cross_entropy",
        "state_fixed_nnz_metadata": metadata,
        "architecture": {
            "inputs": ["state_current_fixed_nnz", "expert_transition_probabilities"],
            "state_hidden_units": args.state_hidden_units,
            "fusion_hidden_units": args.fusion_hidden_units,
            "expert_transition": "direct_gated_residual",
        },
        "hyperparameters": {
            "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
            "l2_regularization": args.l2_regularization, "batch_size": args.batch_size,
            "shuffle_block_size": args.shuffle_block_size, "epochs_max": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "model_path": str(model_path), "best_weights_path": str(best_path),
    })
    print(f"Build 7 minimale state+q completato: validation Recall@6 {recall:.6f}")
    print(f"Modello: {model_path}")


if __name__ == "__main__":
    main()
