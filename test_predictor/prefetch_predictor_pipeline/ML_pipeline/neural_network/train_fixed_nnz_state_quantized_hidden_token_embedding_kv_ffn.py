#!/usr/bin/env python3
"""Train the direct-KV FFN with contiguous fixed-NNZ state feature slots."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from ffn_utils import (
    RANDOM_STATE,
    build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn,
    calculate_recall_at_6,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_materialized_kv_inputs,
    load_quantized_hidden_inputs,
    load_token_inputs,
    make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset,
    parse_hidden_layers,
    require_new_output_directory,
    require_tensorflow,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Addestra la FFN con state fixed-NNZ, hidden int8, token e KV 512D diretto."
    )
    parser.add_argument("--state-data-dir", type=Path, required=True, help="Contiene solo le label y_<split>.npy.")
    parser.add_argument("--fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--kv-data-dir", type=Path, required=True)
    parser.add_argument("--token-data-dir", type=Path, required=True)
    parser.add_argument("--token-embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-weights",
        type=Path,
        default=None,
        help=(
            "Checkpoint di soli pesi da caricare prima dell'addestramento. "
            "Consente di proseguire una rete già addestrata, ma non ripristina "
            "lo stato interno dell'ottimizzatore Adam."
        ),
    )
    parser.add_argument(
        "--initial-epoch",
        type=int,
        default=0,
        help=(
            "Numero di epoche già completate dal checkpoint iniziale. "
            "Con --epochs 1000 e --initial-epoch 20 verranno eseguite le epoche "
            "21..1000, cioè 980 epoche aggiuntive al massimo."
        ),
    )
    parser.add_argument(
        "--state-hidden-units",
        type=int,
        default=None,
        help="Numero di neuroni del Dense del solo ramo state fixed-NNZ (default: 512).",
    )
    parser.add_argument(
        "--fusion-hidden-layers",
        default=None,
        help="Lista separata da virgole dei Dense dopo la concatenazione (default: nessuno).",
    )
    # Compatibilità con i comandi salvati prima della separazione semantica.
    # I nuovi comandi devono usare --state-hidden-units e --fusion-hidden-layers.
    parser.add_argument("--hidden-layers", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.00006997208)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--hidden-projection-dimension", type=int, default=64)
    parser.add_argument("--token-projection-dimension", type=int, default=64)
    parser.add_argument(
        "--raw-hidden-projection-activation",
        choices=("linear", "relu"),
        default="linear",
        help="Attivazione dopo la proiezione raw hidden 4096D -> dimensione scelta (default: linear).",
    )
    parser.add_argument(
        "--token-projection-activation",
        choices=("linear", "relu"),
        default="linear",
        help="Attivazione dopo la proiezione token embedding 4096D -> dimensione scelta (default: linear).",
    )
    parser.add_argument(
        "--kv-projection-dimension",
        choices=("direct", "64", "128", "512"),
        default="direct",
        help="KV diretto 512D oppure Dense(64|128|512) seguito da ReLU (default: direct).",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--shuffle-block-size", type=int, default=65536)
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Usa la policy TensorFlow mixed_float16 per i calcoli compatibili con GPU.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--profile-only-batches", type=int, default=0)
    parser.add_argument("--profile-warmup-batches", type=int, default=3)
    args = parser.parse_args()

    if args.hidden_layers is not None:
        if args.state_hidden_units is not None or args.fusion_hidden_layers is not None:
            parser.error(
                "Usa --state-hidden-units/--fusion-hidden-layers oppure il legacy "
                "--hidden-layers, non entrambi."
            )
        legacy_hidden_layers = parse_hidden_layers(args.hidden_layers, "--hidden-layers")
        state_hidden_units = legacy_hidden_layers[0]
        fusion_hidden_layers = legacy_hidden_layers[1:]
        print(
            "Nota: --hidden-layers è legacy; usa --state-hidden-units e "
            "--fusion-hidden-layers nei nuovi esperimenti."
        )
    else:
        state_hidden_units = 512 if args.state_hidden_units is None else args.state_hidden_units
        fusion_hidden_layers = (
            []
            if args.fusion_hidden_layers is None or not args.fusion_hidden_layers.strip()
            else parse_hidden_layers(args.fusion_hidden_layers, "--fusion-hidden-layers")
        )
    kv_projection_dimension = (
        None if args.kv_projection_dimension == "direct" else int(args.kv_projection_dimension)
    )

    require_new_output_directory(args.output_dir)
    if (min(args.batch_size, args.epochs, state_hidden_units, args.hidden_projection_dimension,
            args.token_projection_dimension) < 1
            or args.shuffle_block_size < args.batch_size
            or args.early_stopping_patience < 0
            or args.profile_only_batches < 0
            or args.profile_warmup_batches < 0
            or args.initial_epoch < 0
            or args.initial_epoch >= args.epochs):
        raise ValueError("Parametri di training non validi.")
    if args.initial_weights is not None and not args.initial_weights.is_file():
        raise FileNotFoundError(f"Checkpoint iniziale non trovato: {args.initial_weights}")

    embeddings_path = args.token_embedding_dir / "token_embeddings_f16.npy"
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embedding token non trovati: {embeddings_path}")
    tf = require_tensorflow()
    compute_policy = "mixed_float16" if args.mixed_precision else "float32"
    tf.keras.mixed_precision.set_global_policy(compute_policy)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("1) Carico state fixed-NNZ, hidden int8, token e KV 512D normalizzati...")
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
    train_token_indices, train_positions = load_token_inputs(
        args.token_data_dir, "train", y_train.shape[0]
    )
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
        raise ValueError("Train e validation usano scale di dequantizzazione diverse.")
    train_kv = load_materialized_kv_inputs(args.kv_data_dir, "train", y_train.shape[0])
    validation_kv = load_materialized_kv_inputs(args.kv_data_dir, "validation", y_validation.shape[0])
    frozen_embeddings = np.load(embeddings_path, mmap_mode="r")
    print(
        f"   train state: indici={train_state_indices.shape} uint16, "
        f"valori={train_state_values.shape} float32, y={y_train.shape}; "
        f"hidden={train_hidden.shape} int8; KV={train_kv.shape} float16"
    )
    print(
        f"   validation: state={validation_state_indices.shape}; KV={validation_kv.shape}; "
        f"scale={train_scale:.6g}"
    )

    print(
        "2) Creo la FFN con somma pesata fixed-NNZ equivalente al Dense state sparse "
        f"(policy={compute_policy})..."
    )
    model = build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn(
        state_metadata["feature_count"], state_metadata["input_slots"], frozen_embeddings,
        state_hidden_units, fusion_hidden_layers, args.activation, args.dropout,
        args.learning_rate, args.l2_regularization, tf,
        args.hidden_projection_dimension, args.token_projection_dimension,
        args.raw_hidden_projection_activation, args.token_projection_activation,
        kv_projection_dimension, "relu", train_scale,
    )
    if args.initial_weights is not None:
        # Il checkpoint Hyperband contiene soltanto le variabili della rete.
        # I pesi vengono quindi recuperati esattamente, mentre Adam ricrea i
        # propri accumulatori e riparte con uno stato dell'ottimizzatore nuovo.
        model.load_weights(args.initial_weights)
        print(
            "   Carico i pesi iniziali dal checkpoint: "
            f"{args.initial_weights} (epoche già completate: {args.initial_epoch})."
        )
    model.summary()
    train_steps = math.ceil(y_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / args.batch_size)
    train_dataset = make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset(
        train_state_indices, train_state_values, y_train, train_token_indices, train_positions,
        train_hidden, train_kv, args.batch_size, True, tf, args.shuffle_block_size,
    ).repeat()

    if args.profile_only_batches:
        args.output_dir.mkdir(parents=True)
        profile_dir = args.output_dir / "tensorboard_profile"
        iterator = iter(train_dataset)
        print(
            f"3) Profilo {args.profile_only_batches} batch fixed-NNZ dopo "
            f"{args.profile_warmup_batches} batch di warm-up..."
        )
        for _ in range(args.profile_warmup_batches):
            inputs, labels = next(iterator)
            model.train_on_batch(inputs, labels)
        options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=2, python_tracer_level=1, device_tracer_level=1
        )
        durations: list[float] = []
        tf.profiler.experimental.start(str(profile_dir), options=options)
        try:
            for _ in range(args.profile_only_batches):
                started_at = time.perf_counter()
                inputs, labels = next(iterator)
                model.train_on_batch(inputs, labels)
                durations.append(time.perf_counter() - started_at)
        finally:
            tf.profiler.experimental.stop()
        save_json(args.output_dir / "reports" / "profile_report.json", {
            "experiment": "fixed_nnz_state_direct_kv_ffn_profile",
            "fixed_nnz_state_data_dir": str(args.fixed_nnz_state_data_dir),
            "tensorflow_compute_policy": compute_policy,
            "batch_size": args.batch_size,
            "profile_only_batches": args.profile_only_batches,
            "profile_warmup_batches": args.profile_warmup_batches,
            "tensorboard_profile_dir": str(profile_dir),
            "end_to_end_batch_seconds": {
                "mean": float(np.mean(durations)), "median": float(np.median(durations)),
                "min": float(np.min(durations)), "max": float(np.max(durations)),
            },
        })
        print("\nProfiling fixed-NNZ completato senza validation/test o salvataggio modello.")
        print(f"Trace TensorBoard: {profile_dir}")
        print(f"Tempo medio end-to-end per batch: {np.mean(durations):.3f}s")
        return

    validation_dataset = make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset(
        validation_state_indices, validation_state_values, y_validation,
        validation_token_indices, validation_positions, validation_hidden, validation_kv,
        args.batch_size, False, tf, args.shuffle_block_size,
    )
    models_dir, reports_dir = args.output_dir / "models", args.output_dir / "reports"
    models_dir.mkdir(parents=True)

    # Misuriamo il checkpoint prima di riprendere. Questo valore entra nel
    # confronto del migliore modello, quindi una continuazione sfavorevole non
    # può sovrascrivere silenziosamente il checkpoint di partenza.
    initial_scores = model.predict(
        validation_dataset.repeat(), steps=validation_steps, verbose=0
    )
    initial_validation_recall = calculate_recall_at_6(initial_scores, y_validation)
    best_weights_path = models_dir / "best.weights.h5"

    class KeepBestWeightsIncludingInitial(tf.keras.callbacks.Callback):
        """Salva e ripristina il massimo Recall@6 includendo il checkpoint iniziale."""

        def __init__(self, initial_score: float, path: Path) -> None:
            super().__init__()
            self.best_score = float(initial_score)
            self.path = path

        def on_train_begin(self, logs=None) -> None:
            self.model.save_weights(self.path)

        def on_epoch_end(self, epoch, logs=None) -> None:
            current_score = None if logs is None else logs.get("val_recall_at_6")
            if current_score is not None and float(current_score) > self.best_score:
                self.best_score = float(current_score)
                self.model.save_weights(self.path)

        def on_train_end(self, logs=None) -> None:
            self.model.load_weights(self.path)

    keep_best_weights = KeepBestWeightsIncludingInitial(
        initial_validation_recall, best_weights_path
    )
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_recall_at_6", mode="max", patience=args.early_stopping_patience,
        restore_best_weights=True,
    ), keep_best_weights]
    print(
        f"3) Addestro con state fixed-NNZ e macro-block shuffle ({args.shuffle_block_size} esempi)..."
    )
    print(f"   Recall@6 validation del checkpoint iniziale: {initial_validation_recall:.6f}")
    started_at = time.perf_counter()
    history = model.fit(
        train_dataset, steps_per_epoch=train_steps,
        validation_data=validation_dataset.repeat(), validation_steps=validation_steps,
        epochs=args.epochs, initial_epoch=args.initial_epoch,
        callbacks=callbacks, shuffle=False, verbose=2,
    )
    elapsed_seconds = time.perf_counter() - started_at
    print("4) Calcolo Recall@6 finale sul validation set...")
    scores = model.predict(validation_dataset.repeat(), steps=validation_steps, verbose=0)
    validation_recall = calculate_recall_at_6(scores, y_validation)
    print(f"   validation Recall@6: {validation_recall:.4f}")
    model_path = models_dir / "ffn.keras"
    model.save(model_path)
    save_json(reports_dir / "training_report.json", {
        "experiment": "fixed_nnz_state_quantized_raw_hidden_token_embedding_direct_kv_ffn_training",
        "state_data_dir": str(args.state_data_dir),
        "fixed_nnz_state_data_dir": str(args.fixed_nnz_state_data_dir),
        "quantized_hidden_data_dir": str(args.quantized_hidden_data_dir),
        "kv_data_dir": str(args.kv_data_dir),
        "token_data_dir": str(args.token_data_dir),
        "token_embedding_dir": str(args.token_embedding_dir),
        "model_path": str(model_path),
        "best_weights_path": str(best_weights_path),
        "initial_weights_path": None if args.initial_weights is None else str(args.initial_weights),
        "initial_epoch": args.initial_epoch,
        "initial_validation_recall_at_6": initial_validation_recall,
        "state_fixed_nnz_metadata": state_metadata,
        "hidden_dequantization_scale": train_scale,
        "tensorflow_compute_policy": compute_policy,
        "kv_compute_policy": "contiguous train-normalized float16; direct 512D concatenation",
        "projection_dimensions": {
            "raw_hidden": args.hidden_projection_dimension,
            "token_embedding": args.token_projection_dimension,
            "kv_current": kv_projection_dimension,
        },
        "projection_activations": {
            "raw_hidden": args.raw_hidden_projection_activation,
            "token_embedding": args.token_projection_activation,
            "kv_current": "relu" if kv_projection_dimension is not None else None,
        },
        "hyperparameters": {
            "state_hidden_units": state_hidden_units,
            "fusion_hidden_layers": fusion_hidden_layers,
            "activation": args.activation,
            "dropout": args.dropout, "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization, "batch_size": args.batch_size,
            "shuffle_block_size": args.shuffle_block_size, "epochs_requested": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "epochs_completed": len(history.history["loss"]),
        "validation_recall_at_6": validation_recall,
        "elapsed_seconds": elapsed_seconds,
        "history": history.history,
    })
    print("\nTraining FFN state fixed-NNZ + hidden int8 + token embedding + KV diretto completato.")
    print(f"Modello: {model_path}")


if __name__ == "__main__":
    main()
