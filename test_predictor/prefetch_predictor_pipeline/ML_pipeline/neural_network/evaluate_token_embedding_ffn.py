#!/usr/bin/env python3
"""Evaluate the frozen-GGUF-token-embedding FFN on the untouched test split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ffn_utils import calculate_recall_at_6, load_split, load_token_inputs, make_token_embedding_dataset, require_tensorflow, save_json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valuta sul test la FFN con embedding token GGUF.")
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--token-data-dir", type=Path, required=True)
    parser.add_argument("--token-embedding-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size deve essere positivo.")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Modello non trovato: {args.model_path}")
    if args.report_path.exists():
        raise FileExistsError(f"Il report esiste gia': {args.report_path}")
    embeddings = np.load(args.token_embedding_dir / "token_embeddings_f16.npy", mmap_mode="r")
    tf = require_tensorflow()
    print("1) Carico test set, input token e modello...")
    x_test, y_test = load_split(args.state_data_dir, "test")
    indices, positions = load_token_inputs(args.token_data_dir, "test", x_test.shape[0])
    if np.any(indices >= len(embeddings)):
        raise ValueError("Indice token test fuori dal lookup.")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    model.compile(jit_compile=False)
    print(f"   test state: X={x_test.shape}, y={y_test.shape}")
    print("2) Calcolo Recall@6 sul test set...")
    steps = math.ceil(x_test.shape[0] / args.batch_size)
    dataset = make_token_embedding_dataset(x_test, y_test, indices, positions, args.batch_size, False, tf).repeat()
    scores = model.predict(dataset, steps=steps, verbose=0)
    recall = calculate_recall_at_6(scores, y_test)
    print(f"   test Recall@6: {recall:.4f}")
    save_json(args.report_path, {
        "experiment": "token_embedding_ffn_test_evaluation", "state_data_dir": str(args.state_data_dir),
        "token_data_dir": str(args.token_data_dir), "token_embedding_dir": str(args.token_embedding_dir),
        "model_path": str(args.model_path), "examples": x_test.shape[0], "batch_size": args.batch_size,
        "test_recall_at_6": recall,
    })
    print(f"\nReport: {args.report_path}")


if __name__ == "__main__":
    main()
