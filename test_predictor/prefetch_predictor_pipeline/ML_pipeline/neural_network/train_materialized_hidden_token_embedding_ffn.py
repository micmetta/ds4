#!/usr/bin/env python3
"""Train the raw-hidden plus token FFN from contiguous normalized hidden arrays."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from ffn_utils import (
    RANDOM_STATE, build_raw_hidden_token_embedding_ffn, calculate_recall_at_6,
    load_materialized_hidden_inputs, load_split, load_token_inputs,
    make_materialized_hidden_token_embedding_dataset, parse_hidden_layers,
    require_new_output_directory, require_tensorflow, save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Addestra la FFN da hidden raw normalizzati materializzati.")
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--materialized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--token-data-dir", type=Path, required=True)
    parser.add_argument("--token-embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-layers", default="512")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.00006997208)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--hidden-projection-dimension", type=int, default=64)
    parser.add_argument("--token-projection-dimension", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--shuffle-block-size", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    args = parser.parse_args()
    require_new_output_directory(args.output_dir)
    if (min(args.batch_size, args.epochs, args.hidden_projection_dimension, args.token_projection_dimension) < 1
            or args.shuffle_block_size < args.batch_size or args.early_stopping_patience < 0):
        raise ValueError("Batch, epoch, patience o dimensioni di proiezione non validi.")
    embeddings_path = args.token_embedding_dir / "token_embeddings_f16.npy"
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embedding token non trovati: {embeddings_path}")
    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("1) Carico state, token e hidden normalizzati materializzati...")
    x_train, y_train = load_split(args.state_data_dir, "train")
    x_validation, y_validation = load_split(args.state_data_dir, "validation")
    train_token_indices, train_positions = load_token_inputs(args.token_data_dir, "train", x_train.shape[0])
    validation_token_indices, validation_positions = load_token_inputs(args.token_data_dir, "validation", x_validation.shape[0])
    train_hidden = load_materialized_hidden_inputs(args.materialized_hidden_data_dir, "train", x_train.shape[0])
    validation_hidden = load_materialized_hidden_inputs(args.materialized_hidden_data_dir, "validation", x_validation.shape[0])
    frozen_embeddings = np.load(embeddings_path, mmap_mode="r")
    print(f"   train state: X={x_train.shape}, y={y_train.shape}; hidden={train_hidden.shape} float16")
    print(f"   validation state: X={x_validation.shape}, y={y_validation.shape}; token lookup={frozen_embeddings.shape}")

    print("2) Creo la FFN multi-input...")
    model = build_raw_hidden_token_embedding_ffn(
        x_train.shape[1], frozen_embeddings, parse_hidden_layers(args.hidden_layers),
        args.activation, args.dropout, args.learning_rate, args.l2_regularization, tf,
        args.hidden_projection_dimension, args.token_projection_dimension,
    )
    model.summary()
    train_steps = math.ceil(x_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(x_validation.shape[0] / args.batch_size)
    train_dataset = make_materialized_hidden_token_embedding_dataset(
        x_train, y_train, train_token_indices, train_positions, train_hidden, args.batch_size, True, tf,
        args.shuffle_block_size,
    ).repeat()
    validation_dataset = make_materialized_hidden_token_embedding_dataset(
        x_validation, y_validation, validation_token_indices, validation_positions, validation_hidden, args.batch_size, False, tf,
    )
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_recall_at_6", mode="max", patience=args.early_stopping_patience, restore_best_weights=True)]
    print(f"3) Addestro la FFN con macro-block shuffle riproducibile ({args.shuffle_block_size} esempi)...")
    started_at = time.perf_counter()
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=validation_dataset.repeat(),
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )
    elapsed_seconds = time.perf_counter() - started_at
    print("4) Calcolo Recall@6 finale sul validation set...")
    scores = model.predict(validation_dataset.repeat(), steps=validation_steps, verbose=0)
    validation_recall = calculate_recall_at_6(scores, y_validation)
    print(f"   validation Recall@6: {validation_recall:.4f}")
    models_dir, reports_dir = args.output_dir / "models", args.output_dir / "reports"
    models_dir.mkdir(parents=True)
    model_path = models_dir / "ffn.keras"
    model.save(model_path)
    save_json(reports_dir / "training_report.json", {
        "experiment": "materialized_raw_hidden_token_embedding_ffn_training",
        "state_data_dir": str(args.state_data_dir), "materialized_hidden_data_dir": str(args.materialized_hidden_data_dir),
        "token_data_dir": str(args.token_data_dir), "token_embedding_dir": str(args.token_embedding_dir), "model_path": str(model_path),
        "state_input_features": x_train.shape[1], "hidden_storage": "contiguous normalized float16, macro-block-shuffled order",
        "raw_hidden_compute_policy": "mixed_float16 with float32 master weights",
        "projection_dimensions": {"raw_hidden": args.hidden_projection_dimension, "token_embedding": args.token_projection_dimension},
        "hyperparameters": {"hidden_layers": parse_hidden_layers(args.hidden_layers), "activation": args.activation, "dropout": args.dropout, "learning_rate": args.learning_rate, "l2_regularization": args.l2_regularization, "batch_size": args.batch_size, "shuffle_block_size": args.shuffle_block_size, "epochs_requested": args.epochs, "early_stopping_patience": args.early_stopping_patience},
        "epochs_completed": len(history.history["loss"]), "validation_recall_at_6": validation_recall, "elapsed_seconds": elapsed_seconds, "history": history.history,
    })
    print("\nTraining FFN hidden materializzato + token embedding completato.")
    print(f"Modello: {model_path}")


if __name__ == "__main__":
    main()
