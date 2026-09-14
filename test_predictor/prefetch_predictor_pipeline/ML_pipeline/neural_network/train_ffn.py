#!/usr/bin/env python3
"""Train one configurable sparse-input feed-forward network with Keras."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from ffn_utils import (
    RANDOM_STATE,
    build_ffn,
    calculate_recall_at_6,
    load_split,
    make_sparse_dataset,
    parse_hidden_layers,
    require_new_output_directory,
    require_tensorflow,
    save_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_DIR = PIPELINE_ROOT / "build_dataset_2" / "ml" / "classical_ml" / "arrays"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "build_dataset_2" / "ml" / "neural_network" / "ffn_default"


def parse_arguments() -> argparse.Namespace:
    """
    Read paths and FFN hyperparameters from the command line.

    Returns:
        Namespace containing one reproducible training configuration.
    """
    parser = argparse.ArgumentParser(description="Addestra una FFN Keras per il prefetch predictor.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-layers", default="512,256")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """
    Train one FFN and save its Keras model, history and validation metrics.

    Returns:
        None. Writes all results below ``--output-dir``.

    Raises:
        ValueError: If one command-line hyperparameter is invalid.
    """
    args = parse_arguments()
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    require_new_output_directory(args.output_dir)
    if args.epochs < 1 or args.early_stopping_patience < 0:
        raise ValueError("--epochs deve essere positivo e patience non negativa.")

    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("1) Carico matrici sparse e label...")
    x_train, y_train = load_split(args.data_dir, "train")
    x_validation, y_validation = load_split(args.data_dir, "validation")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("Train e validation devono avere lo stesso numero di feature.")
    print(f"   train: X={x_train.shape}, y={y_train.shape}")
    print(f"   validation: X={x_validation.shape}, y={y_validation.shape}")

    print("2) Creo la FFN...")
    model = build_ffn(
        input_features=x_train.shape[1],
        hidden_layers=hidden_layers,
        activation=args.activation,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        tf=tf,
    )
    model.summary()

    train_steps = math.ceil(x_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(x_validation.shape[0] / args.batch_size)
    train_dataset = make_sparse_dataset(
        x_train, y_train, args.batch_size, shuffle=True, tf=tf
    ).repeat()
    validation_dataset = make_sparse_dataset(
        x_validation, y_validation, args.batch_size, shuffle=False, tf=tf
    )
    validation_training_dataset = validation_dataset.repeat()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall_at_6",
            mode="max",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
        )
    ]

    print("3) Addestro la FFN...")
    started_at = time.perf_counter()
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=validation_training_dataset,
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )
    elapsed_seconds = time.perf_counter() - started_at

    print("4) Calcolo Recall@6 finale sul validation set...")
    validation_scores = model.predict(
        validation_dataset.repeat(),
        steps=validation_steps,
        verbose=0,
    )
    validation_recall = calculate_recall_at_6(validation_scores, y_validation)
    print(f"   validation Recall@6: {validation_recall:.4f}")

    models_dir = args.output_dir / "models"
    reports_dir = args.output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "ffn.keras")
    save_json(
        reports_dir / "training_report.json",
        {
            "experiment": "ffn_training",
            "data_dir": str(args.data_dir),
            "model_path": str(models_dir / "ffn.keras"),
            "input_features": x_train.shape[1],
            "outputs": y_train.shape[1],
            "hyperparameters": {
                "hidden_layers": hidden_layers,
                "activation": args.activation,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "l2_regularization": args.l2_regularization,
                "batch_size": args.batch_size,
                "epochs_requested": args.epochs,
                "early_stopping_patience": args.early_stopping_patience,
            },
            "epochs_completed": len(history.history["loss"]),
            "validation_recall_at_6": validation_recall,
            "elapsed_seconds": elapsed_seconds,
            "history": history.history,
        },
    )
    print("\nTraining FFN completato.")
    print(f"Modello: {models_dir / 'ffn.keras'}")
    print(f"Report: {reports_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
