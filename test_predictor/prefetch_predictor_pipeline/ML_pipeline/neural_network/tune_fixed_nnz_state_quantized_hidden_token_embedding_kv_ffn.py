#!/usr/bin/env python3
"""Hyperband tuning for the Build 5 fixed-NNZ multi-input FFN."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ffn_utils import (
    RANDOM_STATE,
    build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_materialized_kv_inputs,
    load_quantized_hidden_inputs,
    load_token_inputs,
    make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset,
    require_tensorflow,
)


LEARNING_RATES = (0.00003, 0.00005, 0.00006997208, 0.0001, 0.00015)
PROJECTION_DIMENSIONS = (64, 512, 1024)
KV_PROJECTION_CHOICES = ("direct", "64", "128", "512")


def write_json_atomically(path: Path, content: Any) -> None:
    """Write one JSON snapshot without leaving a partially written target file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(content, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def parse_arguments() -> argparse.Namespace:
    """Read the data locations, Hyperband budget and resume choice."""
    parser = argparse.ArgumentParser(
        description="Cerca con Hyperband le varianti Build 5 fixed-NNZ, hidden, token e KV."
    )
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--kv-data-dir", type=Path, required=True)
    parser.add_argument("--token-data-dir", type=Path, required=True)
    parser.add_argument("--token-embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Riprende un tuning interrotto nella stessa output-dir.")
    parser.add_argument("--state-hidden-units", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--shuffle-block-size", type=int, default=65536)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--hyperband-iterations", type=int, default=1)
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid budgets before data or TensorFlow are loaded."""
    if (
        min(
            args.state_hidden_units,
            args.batch_size,
            args.shuffle_block_size,
            args.max_epochs,
            args.factor,
            args.hyperband_iterations,
        ) < 1
        or args.shuffle_block_size < args.batch_size
        or args.early_stopping_patience < 0
        or args.dropout < 0
        or args.dropout >= 1
        or args.l2_regularization < 0
    ):
        raise ValueError("Parametri del tuning non validi.")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"La cartella output esiste gia': {args.output_dir}. Usa --resume per "
            "riprendere quel tuning oppure scegli una nuova cartella."
        )
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError(f"Non esiste alcun tuning da riprendere: {args.output_dir}")


def main() -> None:
    """Run an interruptible Hyperband search and persist every trial result."""
    args = parse_arguments()
    validate_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    search_configuration = {
        "data": {
            "state_data_dir": str(args.state_data_dir),
            "fixed_nnz_state_data_dir": str(args.fixed_nnz_state_data_dir),
            "quantized_hidden_data_dir": str(args.quantized_hidden_data_dir),
            "kv_data_dir": str(args.kv_data_dir),
            "token_data_dir": str(args.token_data_dir),
            "token_embedding_dir": str(args.token_embedding_dir),
        },
        "search_space": {
            "raw_hidden_projection_dimension": list(PROJECTION_DIMENSIONS),
            "token_projection_dimension": list(PROJECTION_DIMENSIONS),
            "kv_projection": list(KV_PROJECTION_CHOICES),
            "fusion_hidden_units": [0, 512],
            "learning_rate": list(LEARNING_RATES),
            "raw_hidden_projection_activation": "relu (fissa)",
            "token_projection_activation": "relu (fissa)",
        },
        "fixed_hyperparameters": {
            "state_hidden_units": args.state_hidden_units,
            "activation": "relu",
            "dropout": args.dropout,
            "l2_regularization": args.l2_regularization,
            "batch_size": args.batch_size,
            "shuffle_block_size": args.shuffle_block_size,
        },
        "hyperband": {
            "max_epochs": args.max_epochs,
            "factor": args.factor,
            "hyperband_iterations": args.hyperband_iterations,
            "early_stopping_patience": args.early_stopping_patience,
            "objective": "val_recall_at_6 (massimizzare)",
        },
    }
    configuration_path = args.output_dir / "reports" / "search_configuration.json"
    if args.resume:
        if not configuration_path.is_file():
            raise FileNotFoundError(
                "Manca search_configuration.json: non e' sicuro riprendere questa cartella."
            )
        previous_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        if previous_configuration != search_configuration:
            raise ValueError(
                "La configurazione richiesta differisce da quella salvata. Per non "
                "mescolare trial incompatibili, riprendi con gli stessi parametri."
            )
    else:
        write_json_atomically(configuration_path, search_configuration)

    try:
        import keras_tuner as kt
    except ImportError as error:
        raise RuntimeError("KerasTuner non e' installato. Esegui 'uv sync' in ML_pipeline.") from error

    tf = require_tensorflow()
    # La mixed precision globale e' stata misurata come piu' lenta su questa GPU.
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    embeddings_path = args.token_embedding_dir / "token_embeddings_f16.npy"
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embedding token non trovati: {embeddings_path}")

    print("1) Carico gli array Build 5 per train e validation...")
    y_train = load_labels(args.state_data_dir, "train")
    y_validation = load_labels(args.state_data_dir, "validation")
    train_state_indices, train_state_values, state_metadata = load_fixed_nnz_state_inputs(
        args.fixed_nnz_state_data_dir, "train", y_train.shape[0]
    )
    validation_state_indices, validation_state_values, validation_metadata = load_fixed_nnz_state_inputs(
        args.fixed_nnz_state_data_dir, "validation", y_validation.shape[0]
    )
    if state_metadata != validation_metadata:
        raise ValueError("Train e validation usano metadata state fixed-NNZ diverse.")
    train_token_indices, train_positions = load_token_inputs(args.token_data_dir, "train", y_train.shape[0])
    validation_token_indices, validation_positions = load_token_inputs(
        args.token_data_dir, "validation", y_validation.shape[0]
    )
    train_hidden, train_scale = load_quantized_hidden_inputs(
        args.quantized_hidden_data_dir, "train", y_train.shape[0]
    )
    validation_hidden, validation_scale = load_quantized_hidden_inputs(
        args.quantized_hidden_data_dir, "validation", y_validation.shape[0]
    )
    if train_scale != validation_scale:
        raise ValueError("Train e validation usano scale hidden int8 diverse.")
    train_kv = load_materialized_kv_inputs(args.kv_data_dir, "train", y_train.shape[0])
    validation_kv = load_materialized_kv_inputs(args.kv_data_dir, "validation", y_validation.shape[0])
    frozen_embeddings = np.load(embeddings_path, mmap_mode="r")

    train_steps = math.ceil(y_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / args.batch_size)
    train_dataset = make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset(
        train_state_indices, train_state_values, y_train, train_token_indices, train_positions,
        train_hidden, train_kv, args.batch_size, True, tf, args.shuffle_block_size,
    ).repeat()
    validation_dataset = make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset(
        validation_state_indices, validation_state_values, y_validation,
        validation_token_indices, validation_positions, validation_hidden, validation_kv,
        args.batch_size, False, tf, args.shuffle_block_size,
    ).repeat()

    progress_directory = args.output_dir / "trial_progress"
    checkpoint_directory = args.output_dir / "trial_checkpoints"

    class TrialProgressCallback(tf.keras.callbacks.Callback):
        """Append each completed epoch so Ctrl+C preserves partial trial evidence."""

        def __init__(self, trial_id: str, hyperparameters: dict[str, Any]) -> None:
            super().__init__()
            self.trial_id = trial_id
            self.hyperparameters = hyperparameters
            self.path = progress_directory / f"{trial_id}.jsonl"
            self.epoch_started_at: float | None = None

        def on_epoch_begin(self, epoch, logs=None) -> None:
            self.epoch_started_at = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None) -> None:
            logs = logs or {}
            record = {
                "trial_id": self.trial_id,
                "epoch": int(epoch) + 1,
                "hyperparameters": self.hyperparameters,
                "epoch_seconds": (
                    None
                    if self.epoch_started_at is None
                    else time.perf_counter() - self.epoch_started_at
                ),
                "metrics": {name: float(value) for name, value in logs.items()},
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()

    class PersistentHyperband(kt.Hyperband):
        """Hyperband that writes independent progress and checkpoint files per trial."""

        def write_snapshot(self) -> None:
            trials = []
            for trial in sorted(self.oracle.trials.values(), key=lambda item: item.trial_id):
                trials.append(
                    {
                        "trial_id": trial.trial_id,
                        "status": trial.status,
                        "score": trial.score,
                        "best_step": trial.best_step,
                        "hyperparameters": trial.hyperparameters.values,
                        "state": trial.get_state(),
                        "epoch_progress_path": str(progress_directory / f"{trial.trial_id}.jsonl"),
                        "timing_path": str(progress_directory / f"{trial.trial_id}_timing.json"),
                        "checkpoint_path": str(checkpoint_directory / trial.trial_id / "best.weights.h5"),
                    }
                )
            write_json_atomically(
                args.output_dir / "reports" / "trials_snapshot.json",
                {"saved_at_unix_seconds": time.time(), "trials": trials},
            )

        def run_trial(self, trial, *run_args, **run_kwargs):
            callbacks = list(run_kwargs.pop("callbacks", []))
            checkpoint_path = checkpoint_directory / trial.trial_id / "best.weights.h5"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            callbacks.extend(
                [
                    TrialProgressCallback(trial.trial_id, dict(trial.hyperparameters.values)),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=str(checkpoint_path),
                        monitor="val_recall_at_6",
                        mode="max",
                        save_best_only=True,
                        save_weights_only=True,
                    ),
                ]
            )
            run_kwargs["callbacks"] = callbacks
            started_at_unix_seconds = time.time()
            started_at = time.perf_counter()
            try:
                return super().run_trial(trial, *run_args, **run_kwargs)
            finally:
                # Anche un trial fallito o interrotto conserva il tempo già
                # trascorso, utile per confrontare costo e qualità delle run.
                write_json_atomically(
                    progress_directory / f"{trial.trial_id}_timing.json",
                    {
                        "trial_id": trial.trial_id,
                        "started_at_unix_seconds": started_at_unix_seconds,
                        "ended_at_unix_seconds": time.time(),
                        "elapsed_seconds": time.perf_counter() - started_at,
                    },
                )

        def on_trial_end(self, trial):
            result = super().on_trial_end(trial)
            self.write_snapshot()
            return result

    def build_model(hp):
        """Build one candidate: ReLU is fixed, dimensions and fusion are tuned."""
        raw_hidden_dimension = hp.Choice(
            "raw_hidden_projection_dimension", values=PROJECTION_DIMENSIONS
        )
        token_dimension = hp.Choice("token_projection_dimension", values=PROJECTION_DIMENSIONS)
        kv_choice = hp.Choice("kv_projection", values=KV_PROJECTION_CHOICES)
        kv_dimension = None if kv_choice == "direct" else int(kv_choice)
        fusion_units = hp.Choice("fusion_hidden_units", values=(0, 512))
        learning_rate = hp.Choice("learning_rate", values=LEARNING_RATES)
        return build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn(
            state_metadata["feature_count"],
            state_metadata["input_slots"],
            frozen_embeddings,
            args.state_hidden_units,
            [] if fusion_units == 0 else [fusion_units],
            "relu",
            args.dropout,
            learning_rate,
            args.l2_regularization,
            tf,
            raw_hidden_dimension,
            token_dimension,
            "relu",
            "relu",
            kv_dimension,
            "relu",
            train_scale,
        )

    tuner = PersistentHyperband(
        hypermodel=build_model,
        objective=kt.Objective("val_recall_at_6", direction="max"),
        max_epochs=args.max_epochs,
        factor=args.factor,
        hyperband_iterations=args.hyperband_iterations,
        directory=str(args.output_dir / "tuner"),
        project_name="build5_fixed_nnz_hyperband",
        overwrite=False,
        seed=RANDOM_STATE,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=10,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall_at_6",
            mode="max",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
        )
    ]

    print(
        "2) Avvio/riprendo Hyperband: ReLU fissa per raw hidden e token; "
        "salvataggio indipendente dopo ogni trial ed epoca..."
    )
    started_at = time.perf_counter()
    interrupted = False
    try:
        tuner.search(
            train_dataset,
            steps_per_epoch=train_steps,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            epochs=args.max_epochs,
            callbacks=callbacks,
            shuffle=False,
            verbose=2,
        )
    except KeyboardInterrupt:
        interrupted = True
        tuner.write_snapshot()
        write_json_atomically(
            args.output_dir / "reports" / "interruption_report.json",
            {
                "interrupted_at_unix_seconds": time.time(),
                "message": "Tuning interrotto dall'utente con Ctrl+C; i trial completati restano salvati.",
                "resume_command_requires": "lo stesso comando con --resume",
            },
        )
        print("\nTuning interrotto: snapshot scritto. Riprendi con lo stesso comando e --resume.")
    finally:
        tuner.write_snapshot()

    elapsed_seconds = time.perf_counter() - started_at
    completed_trials = [
        trial for trial in tuner.oracle.trials.values() if trial.status == "COMPLETED" and trial.score is not None
    ]
    completed_trials.sort(key=lambda trial: trial.score, reverse=True)
    best_trials = []
    seen_architectures = set()
    for trial in completed_trials:
        hyperparameters = {
            name: value
            for name, value in trial.hyperparameters.values.items()
            if not name.startswith("tuner/")
        }
        architecture_key = json.dumps(hyperparameters, sort_keys=True)
        # Lo stesso candidato puo' riapparire in round Hyperband successivi:
        # per la fase 2 ne vogliamo una sola copia, con lo score migliore.
        if architecture_key in seen_architectures:
            continue
        seen_architectures.add(architecture_key)
        best_trials.append(
            {
                "trial_id": trial.trial_id,
                "validation_recall_at_6": trial.score,
                "hyperparameters": hyperparameters,
                "checkpoint_path": str(checkpoint_directory / trial.trial_id / "best.weights.h5"),
            }
        )
        if len(best_trials) == 5:
            break
    write_json_atomically(
        args.output_dir / "reports" / "tuning_report.json",
        {
            "experiment": "build5_fixed_nnz_hyperband",
            "interrupted": interrupted,
            "elapsed_seconds_this_invocation": elapsed_seconds,
            "completed_trials": len(completed_trials),
            "best_trials_for_phase_2": best_trials,
            "phase_2": {"epochs": 1000, "early_stopping_patience": 5},
        },
    )
    if not interrupted:
        print(f"\nHyperband completato: {len(completed_trials)} trial completati.")
        print(f"Report: {args.output_dir / 'reports' / 'tuning_report.json'}")


if __name__ == "__main__":
    main()
