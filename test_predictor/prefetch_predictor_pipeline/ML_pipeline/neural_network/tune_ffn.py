#!/usr/bin/env python3
"""Tune a sparse-input feed-forward network with KerasTuner RandomSearch."""

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
    require_new_output_directory,
    require_tensorflow,
    save_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_DIR = PIPELINE_ROOT / "build_dataset_2" / "ml" / "classical_ml" / "arrays"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "build_dataset_2" / "ml" / "neural_network" / "ffn_tuning"


def parse_arguments() -> argparse.Namespace:
    """
    Read paths, search ranges and training limits from the command line.

    Returns:
        Namespace containing the RandomSearch configuration.
    """
    parser = argparse.ArgumentParser(description="Cerca iperparametri FFN con KerasTuner RandomSearch.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--max-hidden-layers", type=int, default=3)
    parser.add_argument("--min-units", type=int, default=128)
    parser.add_argument("--max-units", type=int, default=512)
    parser.add_argument("--unit-step", type=int, default=128)
    parser.add_argument("--activations", default="relu,gelu")
    parser.add_argument("--max-dropout", type=float, default=0.4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-learning-rate", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> list[str]:
    """
    Validate tuner ranges and split the activation list.

    Args:
        args: Parsed command-line values.

    Returns:
        Non-empty activation names requested for the search.

    Raises:
        ValueError: If a trial range or training setting is invalid.
    """
    activations = [value.strip() for value in args.activations.split(",") if value.strip()]
    if not activations:
        raise ValueError("--activations deve contenere almeno una attivazione Keras.")
    if (
        args.max_trials < 1
        or args.max_hidden_layers < 1
        or args.min_units < 1
        or args.max_units < args.min_units
        or args.unit_step < 1
        or args.batch_size < 1
        or args.epochs < 1
        or args.early_stopping_patience < 0
    ):
        raise ValueError("I limiti interi del tuning devono essere validi e positivi.")
    if (
        args.max_dropout < 0
        or args.max_dropout >= 1
        or args.min_learning_rate <= 0
        or args.max_learning_rate < args.min_learning_rate
    ):
        raise ValueError("I limiti di dropout e learning rate non sono validi.")
    return activations


def main() -> None:
    """
    Search FFN hyperparameters and train one final model with the best trial.

    Returns:
        None. Saves KerasTuner state, final model and a JSON report.

    Raises:
        RuntimeError: If KerasTuner is missing from the ML environment.
        ValueError: If command-line ranges are invalid.
    """
    args = parse_arguments()
    activations = validate_arguments(args)
    require_new_output_directory(args.output_dir)
    tf = require_tensorflow()
    try:
        import keras_tuner as kt
    except ImportError as error:
        raise RuntimeError("KerasTuner non e' installato. Esegui 'uv sync' in ML_pipeline.") from error
    try:
        from tensorboard.plugins.hparams import api as hparams_api
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard non e' installato. Esegui 'uv sync' in ML_pipeline "
            "prima di lanciare il tuning."
        ) from error
    del hparams_api
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("1) Carico matrici sparse e label...")
    x_train, y_train = load_split(args.data_dir, "train")
    x_validation, y_validation = load_split(args.data_dir, "validation")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("Train e validation devono avere lo stesso numero di feature.")
    train_steps = math.ceil(x_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(x_validation.shape[0] / args.batch_size)
    train_dataset = make_sparse_dataset(
        x_train, y_train, args.batch_size, shuffle=True, tf=tf
    ).repeat()
    validation_dataset = make_sparse_dataset(
        x_validation, y_validation, args.batch_size, shuffle=False, tf=tf
    )
    validation_training_dataset = validation_dataset.repeat()

    def build_model(hp):
        """Build one trial model from the current KerasTuner hyperparameters."""
        number_of_layers = hp.Int("hidden_layers", min_value=1, max_value=args.max_hidden_layers)
        hidden_layers = [
            hp.Int(
                f"units_{index}",
                min_value=args.min_units,
                max_value=args.max_units,
                step=args.unit_step,
            )
            for index in range(1, number_of_layers + 1)
        ]
        return build_ffn(
            input_features=x_train.shape[1],
            hidden_layers=hidden_layers,
            activation=hp.Choice("activation", values=activations),
            dropout=hp.Float("dropout", min_value=0.0, max_value=args.max_dropout, step=0.1),
            learning_rate=hp.Float(
                "learning_rate",
                min_value=args.min_learning_rate,
                max_value=args.max_learning_rate,
                sampling="log",
            ),
            l2_regularization=0.0,
            tf=tf,
        )

    tuner_dir = args.output_dir / "tuner"
    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective=kt.Objective("val_recall_at_6", direction="max"),
        max_trials=args.max_trials,
        overwrite=True,
        directory=str(tuner_dir),
        project_name="random_search",
        seed=RANDOM_STATE,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall_at_6",
            mode="max",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
        )
    ]

    print("2) Cerco gli iperparametri FFN...")
    started_at = time.perf_counter()
    tuner.search(
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

    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    validation_scores = best_model.predict(
        validation_dataset.repeat(),
        steps=validation_steps,
        verbose=0,
    )
    validation_recall = calculate_recall_at_6(validation_scores, y_validation)
    print(f"   migliore validation Recall@6: {validation_recall:.4f}")

    models_dir = args.output_dir / "models"
    reports_dir = args.output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model.save(models_dir / "best_ffn.keras")
    save_json(
        reports_dir / "tuning_report.json",
        {
            "experiment": "ffn_random_search",
            "data_dir": str(args.data_dir),
            "tuner_dir": str(tuner_dir),
            "best_model_path": str(models_dir / "best_ffn.keras"),
            "input_features": x_train.shape[1],
            "outputs": y_train.shape[1],
            "max_trials": args.max_trials,
            "epochs_per_trial": args.epochs,
            "batch_size": args.batch_size,
            "best_hyperparameters": best_hyperparameters.values,
            "validation_recall_at_6": validation_recall,
            "elapsed_seconds": elapsed_seconds,
        },
    )
    print("\nTuning FFN completato.")
    print(f"Modello migliore: {models_dir / 'best_ffn.keras'}")
    print(f"Report: {reports_dir / 'tuning_report.json'}")


if __name__ == "__main__":
    main()
