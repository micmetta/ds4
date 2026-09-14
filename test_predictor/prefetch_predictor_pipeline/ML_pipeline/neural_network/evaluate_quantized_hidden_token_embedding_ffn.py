#!/usr/bin/env python3
"""Evaluate a selected quantized-hidden token-embedding FFN on test only."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ffn_utils import (
    calculate_recall_at_6, load_quantized_hidden_inputs, load_split,
    load_token_inputs, make_quantized_hidden_token_embedding_dataset,
    require_tensorflow, save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valuta sul test la FFN con hidden int8 e token embedding.")
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--token-data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.batch_size < 1 or not args.model_path.is_file():
        raise ValueError("Batch non valido o modello non trovato.")
    if args.report_path.exists():
        raise FileExistsError(f"Il report esiste gia': {args.report_path}")

    tf = require_tensorflow()
    print("1) Carico test state, token, hidden int8 e modello...")
    x_test, y_test = load_split(args.state_data_dir, "test")
    token_indices, positions = load_token_inputs(args.token_data_dir, "test", x_test.shape[0])
    hidden, scale = load_quantized_hidden_inputs(
        args.quantized_hidden_data_dir, "test", x_test.shape[0]
    )
    model = tf.keras.models.load_model(args.model_path, compile=False)
    print("2) Calcolo Recall@6 sul test set...")
    steps = math.ceil(x_test.shape[0] / args.batch_size)
    dataset = make_quantized_hidden_token_embedding_dataset(
        x_test, y_test, token_indices, positions, hidden,
        args.batch_size, False, tf,
    ).repeat()
    scores = model.predict(dataset, steps=steps, verbose=0)
    recall = calculate_recall_at_6(scores, y_test)
    print(f"   test Recall@6: {recall:.4f}")
    save_json(args.report_path, {
        "experiment": "quantized_raw_hidden_token_embedding_ffn_test_evaluation",
        "model_path": str(args.model_path),
        "quantized_hidden_data_dir": str(args.quantized_hidden_data_dir),
        "hidden_dequantization_scale": scale,
        "examples": x_test.shape[0],
        "batch_size": args.batch_size,
        "test_recall_at_6": recall,
    })
    print(f"\nReport: {args.report_path}")


if __name__ == "__main__":
    main()
