#!/usr/bin/env python3
"""Train the Build 6 FFN with current and previous-layer hidden/KV inputs."""

from __future__ import annotations

import argparse
import math
import time
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np

from ffn_utils import (
    RANDOM_STATE,
    build_fixed_nnz_state_two_hidden_two_kv_deeper_ffn,
    build_fixed_nnz_state_two_hidden_two_kv_ffn,
    calculate_recall_at_6,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_materialized_kv_inputs,
    load_quantized_hidden_inputs,
    make_fixed_nnz_state_two_hidden_two_kv_dataset,
    parse_hidden_layers,
    require_new_output_directory,
    require_tensorflow,
    save_json,
)


def load_model_and_adam(path: Path, model) -> None:
    """Restore a promotion checkpoint with both model tensors and Adam state.

    The Build 6 tuner writes ``.model_and_adam.npz`` files at every promotion.
    Restoring Adam as well as the weights makes a subsequent long training run
    continue from the same optimizer state selected by the tuner.
    """
    with np.load(path) as archive:
        model_keys = sorted(
            (key for key in archive.files if key.startswith("model_")),
            key=lambda key: int(key.removeprefix("model_")),
        )
        adam_keys = sorted(
            (key for key in archive.files if key.startswith("adam_")),
            key=lambda key: int(key.removeprefix("adam_")),
        )
        weights = [archive[key] for key in model_keys]
        adam_values = [archive[key] for key in adam_keys]
    model.set_weights(weights)
    # Adam creates momentum/velocity variables lazily.  Create them before
    # assigning the saved tensors so their order and shapes are deterministic.
    model.optimizer.build(model.trainable_variables)
    optimizer_variables = list(model.optimizer.variables)
    if len(adam_values) != len(optimizer_variables):
        raise ValueError("Checkpoint Adam incompatibile con il modello Build 6.")
    for variable, value in zip(optimizer_variables, adam_values, strict=True):
        if tuple(variable.shape) != tuple(value.shape):
            raise ValueError("Checkpoint Adam incompatibile: shape diversa.")
        variable.assign(value)


def load_keras_model_and_adam(path: Path, model) -> None:
    """Restore model tensors and Adam state from a Keras ``.keras`` archive.

    ``model.save`` stores the weights and optimizer variables together in the
    ``model.weights.h5`` member of the archive.  Loading them into the model
    rebuilt from the explicit command-line architecture keeps this path simple
    and avoids relying on deserialisation of implementation-local Keras layers.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        weights_path = Path(temporary_directory) / "model.weights.h5"
        with zipfile.ZipFile(path) as archive:
            with archive.open("model.weights.h5") as source, weights_path.open("wb") as destination:
                destination.write(source.read())
        model.load_weights(weights_path)
        with h5py.File(weights_path, "r") as weights_file:
            optimizer_group = weights_file.get("optimizer/vars")
            if optimizer_group is None:
                raise ValueError("Archivio .keras privo dello stato Adam.")
            adam_values = [
                np.asarray(optimizer_group[name])
                for name in sorted(optimizer_group.keys(), key=int)
            ]
    model.optimizer.build(model.trainable_variables)
    optimizer_variables = list(model.optimizer.variables)
    if len(adam_values) != len(optimizer_variables):
        raise ValueError("Archivio .keras con stato Adam incompatibile con il modello Build 6.")
    for variable, value in zip(optimizer_variables, adam_values, strict=True):
        if tuple(variable.shape) != tuple(value.shape):
            raise ValueError("Archivio .keras con stato Adam di shape incompatibile.")
        variable.assign(value)


def load_positions(directory: Path, split: str, examples: int) -> np.ndarray:
    path = directory / "arrays" / f"token_position_{split}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"token_position non trovata: {path}")
    values = np.load(path, mmap_mode="r")
    if values.shape != (examples, 1) or values.dtype != np.float32:
        raise ValueError(f"token_position incompatibile: {path}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Addestra la FFN Build 6 senza token embedding.")
    parser.add_argument("--state-data-dir", type=Path, required=True, help="Directory con y_<split>.npy.")
    parser.add_argument("--fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--current-kv-data-dir", type=Path, required=True)
    parser.add_argument("--previous-kv-data-dir", type=Path, required=True)
    parser.add_argument("--token-position-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="Checkpoint .model_and_adam.npz prodotto dal tuner.")
    parser.add_argument("--initial-keras", type=Path,
                        help="Archivio .keras di una precedente run, con pesi e Adam.")
    parser.add_argument("--initial-epoch", type=int, default=0,
                        help="Epoche gia' completate nel checkpoint iniziale.")
    parser.add_argument("--initial-best-validation-recall-at-6", type=float,
                        help="Miglior Recall@6 validation gia' raggiunto dal checkpoint iniziale.")
    parser.add_argument("--state-hidden-units", type=int, default=512)
    parser.add_argument("--state-second-units", type=int, default=1024,
                        help="Secondo layer ReLU del ramo state per --architecture deeper_branches.")
    parser.add_argument("--architecture", choices=("compact", "deeper_branches"), default="compact",
                        help="compact conserva la rete migliore attuale; deeper_branches abilita il nuovo esperimento BCE.")
    parser.add_argument("--fusion-hidden-layers", default="512")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.00015)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0,
                        help="Lambda della loss ranking; 0 mantiene la BCE storica.")
    parser.add_argument("--ranking-loss-margin", type=float, default=0.05,
                        help="Margine minimo target/non-target della loss ranking.")
    parser.add_argument("--current-hidden-projection-dimension", type=int, default=512)
    parser.add_argument("--previous-hidden-projection-dimension", type=int, default=512)
    parser.add_argument("--current-kv-projection-dimension",
                        choices=("direct", "64", "128", "256", "512"), default="direct")
    parser.add_argument("--previous-kv-projection-dimension",
                        choices=("direct", "64", "128", "256", "512"), default="direct")
    parser.add_argument("--current-hidden-second-units", type=int, default=128)
    parser.add_argument("--previous-hidden-second-units", type=int, default=32)
    parser.add_argument("--previous-kv-first-units", type=int, default=128)
    parser.add_argument("--previous-kv-second-units", type=int, default=64)
    parser.add_argument("--fusion-second-units", type=int, default=1024)
    parser.add_argument("--branch-dropout", type=float, default=0.1,
                        help="Dropout dopo l'ultimo ReLU dei tre nuovi rami profondi.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--shuffle-block-size", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    # Kept optional so historical Build 6 invocations retain exactly their
    # report identifiers and console output.  Build-specific wrappers set
    # these fields to keep their artefacts self-describing.
    parser.add_argument("--experiment-name", default="build6_fixed_nnz_previous_layer_ffn_training")
    parser.add_argument("--dataset-label", default="Build 6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fusion_hidden_layers = parse_hidden_layers(args.fusion_hidden_layers, "--fusion-hidden-layers") if args.fusion_hidden_layers.strip() else []
    current_kv_projection_dimension = (
        None if args.current_kv_projection_dimension == "direct"
        else int(args.current_kv_projection_dimension)
    )
    previous_kv_projection_dimension = (
        None if args.previous_kv_projection_dimension == "direct"
        else int(args.previous_kv_projection_dimension)
    )
    if (min(args.state_hidden_units, args.current_hidden_projection_dimension,
            args.previous_hidden_projection_dimension, args.batch_size, args.epochs) < 1
            or args.shuffle_block_size < args.batch_size or args.early_stopping_patience < 0):
        raise ValueError("Parametri di training non validi.")
    if args.ranking_loss_weight < 0 or args.ranking_loss_margin <= 0:
        raise ValueError("Iperparametri della ranking loss non validi.")
    if args.architecture == "deeper_branches":
        deeper_dimensions = (
            args.state_second_units,
            args.current_hidden_second_units, args.previous_hidden_second_units,
            args.previous_kv_first_units, args.previous_kv_second_units,
            args.fusion_second_units,
        )
        if (any(value < 1 for value in deeper_dimensions)
                or args.branch_dropout < 0 or args.branch_dropout >= 1):
            raise ValueError("Parametri della FFN deeper_branches non validi.")
        if len(fusion_hidden_layers) != 1:
            raise ValueError("deeper_branches richiede un solo valore in --fusion-hidden-layers: il primo layer di fusione.")
        if current_kv_projection_dimension is not None:
            raise ValueError("deeper_branches mantiene il KV corrente diretto: usare --current-kv-projection-dimension direct.")
        if previous_kv_projection_dimension is not None:
            raise ValueError("deeper_branches usa i due layer KV espliciti: usare --previous-kv-projection-dimension direct.")
        if args.ranking_loss_weight != 0:
            raise ValueError("L'esperimento deeper_branches usa soltanto la BCE classica: impostare --ranking-loss-weight=0.")
    if args.initial_epoch < 0 or args.initial_epoch >= args.epochs:
        raise ValueError("--initial-epoch deve essere >= 0 e minore di --epochs.")
    if args.initial_checkpoint is not None and args.initial_keras is not None:
        raise ValueError("Usare uno solo fra --initial-checkpoint e --initial-keras.")
    if (args.initial_best_validation_recall_at_6 is not None
            and not 0.0 <= args.initial_best_validation_recall_at_6 <= 1.0):
        raise ValueError("--initial-best-validation-recall-at-6 deve essere compreso fra 0 e 1.")
    for initial_path in (args.initial_checkpoint, args.initial_keras):
        if initial_path is not None and not initial_path.is_file():
            raise FileNotFoundError(f"Checkpoint iniziale non trovato: {initial_path}")
    require_new_output_directory(args.output_dir)

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    print(f"1) Carico state {args.dataset_label} fixed-NNZ, i due hidden int8, i due KV e token_position...")
    y_train = load_labels(args.state_data_dir, "train")
    y_validation = load_labels(args.state_data_dir, "validation")
    train_indices, train_values, metadata = load_fixed_nnz_state_inputs(args.fixed_nnz_state_data_dir, "train", y_train.shape[0])
    validation_indices, validation_values, validation_metadata = load_fixed_nnz_state_inputs(args.fixed_nnz_state_data_dir, "validation", y_validation.shape[0])
    if metadata != validation_metadata:
        raise ValueError("Metadata state train e validation diverse.")
    train_position = load_positions(args.token_position_data_dir, "train", y_train.shape[0])
    validation_position = load_positions(args.token_position_data_dir, "validation", y_validation.shape[0])
    train_current_hidden, current_scale = load_quantized_hidden_inputs(args.current_quantized_hidden_data_dir, "train", y_train.shape[0])
    validation_current_hidden, validation_current_scale = load_quantized_hidden_inputs(args.current_quantized_hidden_data_dir, "validation", y_validation.shape[0])
    train_previous_hidden, previous_scale = load_quantized_hidden_inputs(args.previous_quantized_hidden_data_dir, "train", y_train.shape[0])
    validation_previous_hidden, validation_previous_scale = load_quantized_hidden_inputs(args.previous_quantized_hidden_data_dir, "validation", y_validation.shape[0])
    if current_scale != validation_current_scale or previous_scale != validation_previous_scale:
        raise ValueError("Scale hidden train e validation diverse.")
    train_current_kv = load_materialized_kv_inputs(args.current_kv_data_dir, "train", y_train.shape[0])
    validation_current_kv = load_materialized_kv_inputs(args.current_kv_data_dir, "validation", y_validation.shape[0])
    train_previous_kv = load_materialized_kv_inputs(args.previous_kv_data_dir, "train", y_train.shape[0])
    validation_previous_kv = load_materialized_kv_inputs(args.previous_kv_data_dir, "validation", y_validation.shape[0])
    print(f"   train: state={train_indices.shape}; hidden corrente/precedente={train_current_hidden.shape}/{train_previous_hidden.shape}; KV={train_current_kv.shape}/{train_previous_kv.shape}")

    print(f"2) Creo la FFN {args.dataset_label} con rami espliciti per layer corrente e precedente...")
    if args.architecture == "compact":
        model = build_fixed_nnz_state_two_hidden_two_kv_ffn(
            metadata["feature_count"], metadata["input_slots"], args.state_hidden_units,
            fusion_hidden_layers, args.activation, args.dropout, args.learning_rate,
            args.l2_regularization, tf, args.current_hidden_projection_dimension,
            args.previous_hidden_projection_dimension, current_scale, previous_scale,
            current_kv_projection_dimension, previous_kv_projection_dimension,
            args.ranking_loss_weight, args.ranking_loss_margin,
        )
    else:
        model = build_fixed_nnz_state_two_hidden_two_kv_deeper_ffn(
            metadata["feature_count"], metadata["input_slots"], args.state_hidden_units,
            args.state_second_units,
            args.activation, args.dropout, args.branch_dropout, args.dropout,
            args.learning_rate, args.l2_regularization, tf,
            args.current_hidden_projection_dimension, args.current_hidden_second_units,
            args.previous_hidden_projection_dimension, args.previous_hidden_second_units,
            current_scale, previous_scale, args.previous_kv_first_units,
            args.previous_kv_second_units, fusion_hidden_layers[0], args.fusion_second_units,
        )
    if args.initial_checkpoint is not None:
        load_model_and_adam(args.initial_checkpoint, model)
        print(f"   Riprendo pesi e Adam da: {args.initial_checkpoint} "
              f"(epoche gia' completate: {args.initial_epoch}).")
    elif args.initial_keras is not None:
        load_keras_model_and_adam(args.initial_keras, model)
        print(f"   Riprendo pesi e Adam da: {args.initial_keras} "
              f"(epoche gia' completate: {args.initial_epoch}).")
    model.summary()
    train_dataset = make_fixed_nnz_state_two_hidden_two_kv_dataset(
        train_indices, train_values, y_train, train_position, train_current_hidden,
        train_previous_hidden, train_current_kv, train_previous_kv, args.batch_size,
        True, tf, args.shuffle_block_size,
    ).repeat()
    validation_dataset = make_fixed_nnz_state_two_hidden_two_kv_dataset(
        validation_indices, validation_values, y_validation, validation_position,
        validation_current_hidden, validation_previous_hidden, validation_current_kv,
        validation_previous_kv, args.batch_size, False, tf, args.shuffle_block_size,
    )
    train_steps = math.ceil(y_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / args.batch_size)
    models_dir, reports_dir = args.output_dir / "models", args.output_dir / "reports"
    models_dir.mkdir(parents=True)
    best_path = models_dir / "best.weights.h5"
    checkpoint_kwargs = {
        "monitor": "val_recall_at_6",
        "mode": "max",
        "save_best_only": True,
        "save_weights_only": True,
    }
    early_stopping_kwargs = {
        "monitor": "val_recall_at_6",
        "mode": "max",
        "patience": args.early_stopping_patience,
        "restore_best_weights": True,
    }
    if args.initial_best_validation_recall_at_6 is not None:
        # Mantiene come migliore iniziale il checkpoint ripreso, anche se le
        # nuove epoche non riescono a superare il suo Recall@6 validation.
        model.save_weights(best_path)
        checkpoint_kwargs["initial_value_threshold"] = args.initial_best_validation_recall_at_6
        early_stopping_kwargs["baseline"] = args.initial_best_validation_recall_at_6
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_path), **checkpoint_kwargs),
        tf.keras.callbacks.EarlyStopping(**early_stopping_kwargs),
    ]
    print(f"3) Addestro con macro-block shuffle da {args.shuffle_block_size} esempi...")
    started = time.perf_counter()
    history = model.fit(
        train_dataset, steps_per_epoch=train_steps, validation_data=validation_dataset.repeat(),
        validation_steps=validation_steps, initial_epoch=args.initial_epoch,
        epochs=args.epochs, callbacks=callbacks,
        shuffle=False, verbose=2,
    )
    elapsed = time.perf_counter() - started
    model.load_weights(best_path)
    scores = model.predict(validation_dataset.repeat(), steps=validation_steps, verbose=0)
    recall = calculate_recall_at_6(scores, y_validation)
    model_path = models_dir / "ffn.keras"
    model.save(model_path)
    save_json(reports_dir / "training_report.json", {
        "experiment": args.experiment_name,
        "best_validation_recall_at_6": recall,
        "initial_best_validation_recall_at_6": args.initial_best_validation_recall_at_6,
        "elapsed_seconds": elapsed,
        "state_fixed_nnz_metadata": metadata,
        "hidden_dequantization_scales": {"current": current_scale, "previous": previous_scale},
        "architecture": {
            "name": args.architecture, "token_embedding_input": False, "state_hidden_units": args.state_hidden_units,
            "state_second_units": args.state_second_units,
            "current_hidden_projection_dimension": args.current_hidden_projection_dimension,
            "previous_hidden_projection_dimension": args.previous_hidden_projection_dimension,
            "current_kv_projection_dimension": current_kv_projection_dimension,
            "previous_kv_projection_dimension": previous_kv_projection_dimension,
            "fusion_hidden_layers": fusion_hidden_layers,
            "current_hidden_second_units": args.current_hidden_second_units,
            "previous_hidden_second_units": args.previous_hidden_second_units,
            "previous_kv_first_units": args.previous_kv_first_units,
            "previous_kv_second_units": args.previous_kv_second_units,
            "fusion_second_units": args.fusion_second_units,
        },
        "hyperparameters": {"dropout": args.dropout, "branch_dropout": args.branch_dropout, "learning_rate": args.learning_rate, "l2_regularization": args.l2_regularization, "ranking_loss_weight": args.ranking_loss_weight, "ranking_loss_margin": args.ranking_loss_margin, "batch_size": args.batch_size, "shuffle_block_size": args.shuffle_block_size, "initial_epoch": args.initial_epoch, "epochs_max": args.epochs, "early_stopping_patience": args.early_stopping_patience},
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "model_path": str(model_path), "best_weights_path": str(best_path),
    })
    print(f"{args.dataset_label} training completato: validation Recall@6 {recall:.6f}")
    print(f"Modello: {model_path}")


if __name__ == "__main__":
    main()
