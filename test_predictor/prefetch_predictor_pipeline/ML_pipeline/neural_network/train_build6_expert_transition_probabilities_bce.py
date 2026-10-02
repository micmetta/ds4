#!/usr/bin/env python3
"""Continue the best Build 6 transition-probability candidate after tuning.

The successive-halving tuner stops every finalist at epoch 20.  This script
selects its best finalist from ``tuning_report.json``, restores the exact
``.model_and_adam.npz`` promotion checkpoint, then continues the same BCE-only
architecture until early stopping.  It deliberately does not modify the tuner
or any historical model.
"""

from __future__ import annotations

import argparse
import json
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


def load_model_and_adam(path: Path, model) -> None:
    """Restore model weights and the Adam state written by the tuner."""
    with np.load(path) as archive:
        model_values = [
            archive[key] for key in sorted(
                (key for key in archive.files if key.startswith("model_")),
                key=lambda key: int(key.removeprefix("model_")),
            )
        ]
        adam_values = [
            archive[key] for key in sorted(
                (key for key in archive.files if key.startswith("adam_")),
                key=lambda key: int(key.removeprefix("adam_")),
            )
        ]
    model.set_weights(model_values)
    model.optimizer.build(model.trainable_variables)
    optimizer_variables = list(model.optimizer.variables)
    if len(adam_values) != len(optimizer_variables):
        raise ValueError("Checkpoint Adam incompatibile con il modello ricostruito.")
    for variable, value in zip(optimizer_variables, adam_values, strict=True):
        if tuple(variable.shape) != tuple(value.shape):
            raise ValueError("Checkpoint Adam incompatibile: shape diversa.")
        variable.assign(value)


def load_positions(directory: Path, split: str, examples: int) -> np.ndarray:
    path = directory / "arrays" / f"token_position_{split}.npy"
    values = np.load(path, mmap_mode="r")
    if values.shape != (examples, 1) or values.dtype != np.float32:
        raise ValueError(f"token_position incompatibile: {path}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prosegue la migliore candidata BCE con expert_transition_probabilities."
    )
    parser.add_argument("--tuning-report", type=Path, required=True)
    parser.add_argument(
        "--candidate-id", default="best",
        help="ID della finalista, oppure 'best' per la prima del report ordinato.",
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
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="Non ripristina il checkpoint phase_20; serve per ablation architetturali incompatibili.",
    )
    parser.add_argument(
        "--fusion-batch-normalization", action="store_true",
        help="Inserisce BatchNormalization dopo Concatenate e prima della fusione Dense.",
    )
    parser.add_argument(
        "--expert-transition-gated-residual", action="store_true",
        help="Usa q come contributo residuale gated ai 256 logit finali, non nella Concatenate.",
    )
    return parser.parse_args()


def select_candidate(report_path: Path, candidate_id: str) -> tuple[dict, Path, float, int]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    finalists = report.get("finalists_after_20_epochs", [])
    if not finalists:
        raise ValueError("Il report non contiene finaliste phase_20.")
    if candidate_id == "best":
        candidate = finalists[0]
    else:
        candidate = next((item for item in finalists if item["candidate_id"] == candidate_id), None)
        if candidate is None:
            available = ", ".join(item["candidate_id"] for item in finalists)
            raise ValueError(f"Candidata {candidate_id!r} assente. Disponibili: {available}.")
    phase = candidate["phases"]["phase_20"]
    # Derive this path from the report's current root: the saved absolute path
    # may belong to a previous Colab runtime, whereas this root is portable.
    checkpoint = report_path.parent.parent / "candidate_checkpoints" / candidate["candidate_id"] / "phase_20.model_and_adam.npz"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint phase_20 non trovato: {checkpoint}")
    return (
        candidate,
        checkpoint,
        float(phase["best_validation_recall_at_6"]),
        int(phase["target_epochs"]),
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.early_stopping_patience < 0:
        raise ValueError("Servono --epochs >= 1 e patience >= 0.")
    if not args.tuning_report.is_file():
        raise FileNotFoundError(f"Report tuning non trovato: {args.tuning_report}")
    require_new_output_directory(args.output_dir)
    candidate, checkpoint, initial_best, initial_epoch = select_candidate(
        args.tuning_report, args.candidate_id
    )
    if not args.from_scratch and initial_epoch >= args.epochs:
        raise ValueError("--epochs deve essere maggiore delle epoche della phase_20.")
    hyperparameters = candidate["hyperparameters"]
    transition_projection = hyperparameters["expert_transition_projection"]
    transition_dimension = 0 if transition_projection == "direct" else int(transition_projection)
    current_kv_dimension = (
        None if hyperparameters["current_kv_projection"] == "direct"
        else int(hyperparameters["current_kv_projection"])
    )
    previous_kv_dimension = (
        None if hyperparameters["previous_kv_projection"] == "direct"
        else int(hyperparameters["previous_kv_projection"])
    )

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    training_mode = "da zero" if args.from_scratch else f"dal checkpoint phase_20 (epoch {initial_epoch})"
    print(f"1) Seleziono {candidate['candidate_id']} dal report: "
          f"best val Recall@6 phase_20={initial_best:.6f}; training {training_mode}.")
    print("2) Carico i dati Build 6, inclusa la feature 256D di transizione...")
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
    if (current_scale != validation_current_scale or previous_scale != validation_previous_scale):
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

    # The tuning report itself embeds no search_configuration; fixed values are
    # deliberately explicit here to match the completed BCE tuner invocation.
    dropout, l2, batch_size, shuffle_size = 0.3, 0.0, 4096, 65536
    print("3) Ricostruisco la candidata con BCE classica e ripristino pesi + Adam...")
    fusion = [] if hyperparameters["fusion_hidden_units"] == 0 else [hyperparameters["fusion_hidden_units"]]
    model = build_fixed_nnz_state_two_hidden_two_kv_ffn(
        metadata["feature_count"], metadata["input_slots"], hyperparameters["state_hidden_units"],
        fusion, "relu", dropout, hyperparameters["learning_rate"], l2, tf,
        hyperparameters["current_raw_hidden_projection_dimension"],
        hyperparameters["previous_raw_hidden_projection_dimension"], current_scale, previous_scale,
        current_kv_dimension, previous_kv_dimension,
        ranking_loss_weight=0.0, ranking_loss_margin=0.05,
        expert_transition_projection_dimension=transition_dimension,
        fusion_batch_normalization=args.fusion_batch_normalization,
        expert_transition_gated_residual=args.expert_transition_gated_residual,
    )
    if not args.from_scratch:
        load_model_and_adam(checkpoint, model)
        print(f"   Checkpoint: {checkpoint} (ripartenza da epoch {initial_epoch}).")
    else:
        initial_epoch = 0
        print("   Nuova inizializzazione: i pesi della nuova architettura saranno appresi da zero.")
    model.summary()

    train_dataset = make_fixed_nnz_state_two_hidden_two_kv_dataset(
        train_indices, train_values, y_train, train_position, train_current_hidden,
        train_previous_hidden, train_current_kv, train_previous_kv, batch_size, True, tf,
        shuffle_size, train_transition,
    ).repeat()
    validation_dataset = make_fixed_nnz_state_two_hidden_two_kv_dataset(
        validation_indices, validation_values, y_validation, validation_position,
        validation_current_hidden, validation_previous_hidden, validation_current_kv,
        validation_previous_kv, batch_size, False, tf, shuffle_size, validation_transition,
    )
    train_steps = math.ceil(y_train.shape[0] / batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / batch_size)
    models_dir, reports_dir = args.output_dir / "models", args.output_dir / "reports"
    models_dir.mkdir(parents=True)
    best_path = models_dir / "best.weights.h5"
    checkpoint_kwargs = {
        "monitor": "val_recall_at_6", "mode": "max", "save_best_only": True,
        "save_weights_only": True,
    }
    early_stopping_kwargs = {
        "monitor": "val_recall_at_6", "mode": "max",
        "patience": args.early_stopping_patience, "restore_best_weights": True,
    }
    if not args.from_scratch:
        # Preserve the selected tuner checkpoint if continuation cannot improve it.
        model.save_weights(best_path)
        checkpoint_kwargs["initial_value_threshold"] = initial_best
        early_stopping_kwargs["baseline"] = initial_best
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_path), **checkpoint_kwargs),
        tf.keras.callbacks.EarlyStopping(**early_stopping_kwargs),
    ]
    print(f"4) Addestro BCE fino a {args.epochs} epoche con patience={args.early_stopping_patience}...")
    started = time.perf_counter()
    history = model.fit(
        train_dataset, steps_per_epoch=train_steps,
        validation_data=validation_dataset.repeat(), validation_steps=validation_steps,
        initial_epoch=initial_epoch, epochs=args.epochs, callbacks=callbacks,
        shuffle=False, verbose=2,
    )
    elapsed = time.perf_counter() - started
    model.load_weights(best_path)
    scores = model.predict(validation_dataset.repeat(), steps=validation_steps, verbose=0)
    recall = calculate_recall_at_6(scores, y_validation)
    model_path = models_dir / "ffn.keras"
    model.save(model_path)
    save_json(reports_dir / "training_report.json", {
        "experiment": "build6_expert_transition_probabilities_bce_long_training",
        "source_tuning_report": str(args.tuning_report),
        "selected_candidate": candidate,
        "source_checkpoint": str(checkpoint),
        "initial_epoch": initial_epoch,
        "initial_best_validation_recall_at_6": None if args.from_scratch else initial_best,
        "best_validation_recall_at_6": recall,
        "elapsed_seconds": elapsed,
        "loss": "binary_cross_entropy",
        "expert_transition_projection": transition_projection,
        "fusion_batch_normalization": args.fusion_batch_normalization,
        "expert_transition_gated_residual": args.expert_transition_gated_residual,
        "from_scratch": args.from_scratch,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "model_path": str(model_path),
    })
    print(f"Training completato: validation Recall@6 {recall:.6f}")
    print(f"Modello: {model_path}")


if __name__ == "__main__":
    main()
