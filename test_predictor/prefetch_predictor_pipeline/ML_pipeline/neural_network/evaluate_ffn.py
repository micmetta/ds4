#!/usr/bin/env python3
"""Evaluate one saved FFN on the untouched test split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ffn_utils import calculate_recall_at_6, load_split, make_sparse_dataset, require_tensorflow, save_json


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_DIR = PIPELINE_ROOT / "build_dataset_2" / "ml" / "classical_ml" / "arrays"


def parse_arguments() -> argparse.Namespace:
    """
    Read the test dataset, saved model and destination report paths.

    Returns:
        Namespace containing one evaluation configuration.
    """
    parser = argparse.ArgumentParser(description="Valuta una FFN salvata sul test set.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    """
    Load a saved FFN and calculate Recall@6 only on the test split.

    Returns:
        None. Writes one JSON report at ``--report-path``.

    Raises:
        FileExistsError: If the requested test report already exists.
        FileNotFoundError: If the model path or required test files are missing.
        ValueError: If model, test data or batch-size shapes are invalid.
    """
    args = parse_arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size deve essere un intero positivo.")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Modello non trovato: {args.model_path}")
    if args.report_path.exists():
        raise FileExistsError(
            f"Il report esiste gia': {args.report_path}. "
            "Scegli un nuovo --report-path per non sovrascrivere la valutazione."
        )

    tf = require_tensorflow()

    print("1) Carico test set e modello FFN...")
    x_test, y_test = load_split(args.data_dir, "test")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    # Sparse TensorFlow operations used by the first Dense layer are not
    # supported by XLA GPU. The saved model is loaded without its training
    # compilation state, so disable XLA explicitly also for inference.
    model.compile(jit_compile=False)
    if model.input_shape[-1] != x_test.shape[1]:
        raise ValueError(
            f"Il modello richiede {model.input_shape[-1]} feature, "
            f"mentre il test set ne contiene {x_test.shape[1]}."
        )
    print(f"   test: X={x_test.shape}, y={y_test.shape}")

    print("2) Calcolo Recall@6 sul test set...")
    test_steps = math.ceil(x_test.shape[0] / args.batch_size)
    test_dataset = make_sparse_dataset(
        x_test, y_test, args.batch_size, shuffle=False, tf=tf
    ).repeat()
    scores = model.predict(test_dataset, steps=test_steps, verbose=0)
    test_recall = calculate_recall_at_6(scores, y_test)
    print(f"   test Recall@6: {test_recall:.4f}")

    save_json(
        args.report_path,
        {
            "experiment": "ffn_test_evaluation",
            "data_dir": str(args.data_dir),
            "model_path": str(args.model_path),
            "input_features": x_test.shape[1],
            "examples": x_test.shape[0],
            "outputs": y_test.shape[1],
            "batch_size": args.batch_size,
            "test_recall_at_6": test_recall,
        },
    )
    print("\nValutazione FFN completata.")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
