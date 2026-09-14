#!/usr/bin/env python3
"""Evaluate the selected raw-hidden plus token-embedding FFN on test only."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ffn_utils import (
    calculate_recall_at_6, load_raw_hidden_inputs, load_split, load_token_inputs,
    make_raw_hidden_token_embedding_dataset, require_tensorflow, save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valuta sul test la FFN raw hidden + embedding token.")
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--raw-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--token-data-dir", type=Path, required=True)
    parser.add_argument("--token-embedding-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.batch_size < 1 or not args.model_path.is_file():
        raise ValueError("Batch non valido o modello non trovato.")
    if args.report_path.exists():
        raise FileExistsError(f"Il report esiste gia': {args.report_path}")
    embeddings = np.load(args.token_embedding_dir / "token_embeddings_f16.npy", mmap_mode="r")
    tf = require_tensorflow()
    print("1) Carico test set, input token, riferimenti raw hidden e modello...")
    x_test, y_test = load_split(args.state_data_dir, "test")
    token_indices, positions = load_token_inputs(args.token_data_dir, "test", x_test.shape[0])
    hidden_inputs = load_raw_hidden_inputs(args.raw_hidden_data_dir, "test", x_test.shape[0])
    if np.any(token_indices >= len(embeddings)):
        raise ValueError("Indice token test fuori dal lookup.")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    model.compile(jit_compile=False)
    print(f"   test state: X={x_test.shape}, y={y_test.shape}; raw hidden=4096D")
    print("2) Calcolo Recall@6 sul test set...")
    steps = math.ceil(x_test.shape[0] / args.batch_size)
    dataset = make_raw_hidden_token_embedding_dataset(
        x_test, y_test, token_indices, positions, *hidden_inputs, args.batch_size, False, tf,
    ).repeat()
    scores = model.predict(dataset, steps=steps, verbose=0)
    recall = calculate_recall_at_6(scores, y_test)
    print(f"   test Recall@6: {recall:.4f}")
    save_json(args.report_path, {
        "experiment": "raw_hidden_token_embedding_ffn_test_evaluation",
        "state_data_dir": str(args.state_data_dir), "raw_hidden_data_dir": str(args.raw_hidden_data_dir),
        "token_data_dir": str(args.token_data_dir), "token_embedding_dir": str(args.token_embedding_dir),
        "model_path": str(args.model_path), "examples": x_test.shape[0], "batch_size": args.batch_size,
        "test_recall_at_6": recall,
    })
    print(f"\nReport: {args.report_path}")


if __name__ == "__main__":
    main()
