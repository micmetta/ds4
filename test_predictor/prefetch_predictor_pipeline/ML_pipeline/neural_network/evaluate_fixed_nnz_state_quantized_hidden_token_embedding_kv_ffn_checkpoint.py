#!/usr/bin/env python3
"""Evaluate one already-trained fixed-NNZ FFN checkpoint on the untouched test split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ffn_utils import (
    build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn,
    calculate_recall_at_6,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_materialized_kv_inputs,
    load_quantized_hidden_inputs,
    load_token_inputs,
    make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset,
    require_tensorflow,
    save_json,
)


def parse_arguments() -> argparse.Namespace:
    """Read one fixed architecture, its weights and the Build 5 test inputs."""
    parser = argparse.ArgumentParser(
        description="Valuta un checkpoint FFN Build 5 sul solo test set, senza addestrare."
    )
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--kv-data-dir", type=Path, required=True)
    parser.add_argument("--token-data-dir", type=Path, required=True)
    parser.add_argument("--token-embedding-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--state-hidden-units", type=int, required=True)
    parser.add_argument("--fusion-hidden-layers", default="")
    parser.add_argument("--hidden-projection-dimension", type=int, required=True)
    parser.add_argument("--token-projection-dimension", type=int, required=True)
    parser.add_argument("--kv-projection-dimension", choices=("direct", "64", "128", "512"), required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    """Rebuild the selected architecture, load its weights, then score only test data."""
    args = parse_arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {args.checkpoint}")
    if args.output_report.exists():
        raise FileExistsError(f"Il report esiste gia': {args.output_report}")
    if min(args.state_hidden_units, args.hidden_projection_dimension, args.token_projection_dimension, args.batch_size) < 1:
        raise ValueError("Dimensioni o batch size non validi.")
    if args.dropout < 0 or args.dropout >= 1 or args.learning_rate <= 0 or args.l2_regularization < 0:
        raise ValueError("Dropout, learning rate o L2 non validi.")

    fusion_hidden_layers = [int(value) for value in args.fusion_hidden_layers.split(",") if value.strip()]
    if any(value < 1 for value in fusion_hidden_layers):
        raise ValueError("--fusion-hidden-layers deve contenere solo interi positivi.")
    kv_projection_dimension = None if args.kv_projection_dimension == "direct" else int(args.kv_projection_dimension)
    embeddings_path = args.token_embedding_dir / "token_embeddings_f16.npy"
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embedding token non trovati: {embeddings_path}")

    tf = require_tensorflow()
    # Il tuning vincitore ha usato float32; manteniamo identica la policy di calcolo.
    tf.keras.mixed_precision.set_global_policy("float32")

    print("1) Carico esclusivamente gli array del test set Build 5...")
    y_test = load_labels(args.state_data_dir, "test")
    state_indices, state_values, state_metadata = load_fixed_nnz_state_inputs(
        args.fixed_nnz_state_data_dir, "test", y_test.shape[0]
    )
    token_indices, token_positions = load_token_inputs(args.token_data_dir, "test", y_test.shape[0])
    hidden, hidden_scale = load_quantized_hidden_inputs(args.quantized_hidden_data_dir, "test", y_test.shape[0])
    kv = load_materialized_kv_inputs(args.kv_data_dir, "test", y_test.shape[0])
    frozen_embeddings = np.load(embeddings_path, mmap_mode="r")
    print(f"   test: y={y_test.shape}; state={state_indices.shape}; hidden={hidden.shape}; KV={kv.shape}")

    print("2) Ricostruisco l'architettura del trial e carico il suo checkpoint...")
    model = build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn(
        state_metadata["feature_count"], state_metadata["input_slots"], frozen_embeddings,
        args.state_hidden_units, fusion_hidden_layers, "relu", args.dropout,
        args.learning_rate, args.l2_regularization, tf,
        args.hidden_projection_dimension, args.token_projection_dimension,
        "relu", "relu", kv_projection_dimension, "relu", hidden_scale,
    )
    model.load_weights(args.checkpoint)

    test_steps = math.ceil(y_test.shape[0] / args.batch_size)
    test_dataset = make_fixed_nnz_state_quantized_hidden_token_embedding_kv_dataset(
        state_indices, state_values, y_test, token_indices, token_positions,
        hidden, kv, args.batch_size, False, tf, args.batch_size,
    ).repeat()
    print("3) Calcolo le predizioni e Recall@6 sul test set, senza aggiornare alcun peso...")
    scores = model.predict(test_dataset, steps=test_steps, verbose=0)
    test_recall = calculate_recall_at_6(scores, y_test)
    print(f"   test Recall@6: {test_recall:.6f}")

    save_json(args.output_report, {
        "experiment": "fixed_nnz_checkpoint_test_evaluation",
        "split": "test",
        "checkpoint": str(args.checkpoint),
        "test_examples": int(y_test.shape[0]),
        "test_recall_at_6": test_recall,
        "architecture": {
            "state_hidden_units": args.state_hidden_units,
            "fusion_hidden_layers": fusion_hidden_layers,
            "raw_hidden_projection_dimension": args.hidden_projection_dimension,
            "raw_hidden_projection_activation": "relu",
            "token_projection_dimension": args.token_projection_dimension,
            "token_projection_activation": "relu",
            "kv_projection_dimension": kv_projection_dimension,
            "kv_projection_activation": "relu" if kv_projection_dimension is not None else None,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
            "tensorflow_compute_policy": "float32",
        },
    })
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
