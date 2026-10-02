#!/usr/bin/env python3
"""Evaluate the causal 256D transition vector as a no-training baseline.

For every example this baseline predicts the six expert indices with the
largest components of q_{t,l}.  It is intentionally not a neural model: its
Recall@6 quantifies how much of the task is solved by the transition feature
alone before it is fused with the other Build 6 inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NUM_EXPERTS = 256
TOP_K = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline top-6 delle expert transition probabilities.")
    parser.add_argument("--state-data-dir", type=Path, required=True, help="Directory con y_<split>.npy.")
    parser.add_argument(
        "--expert-transition-probabilities-data-dir", type=Path, required=True,
        help="Root della feature; contiene arrays/expert_transition_probabilities_<split>.npy.",
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--splits", default="validation,test", help="Split separati da virgole.")
    return parser.parse_args()


def load_split(state_data_dir: Path, transition_data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    labels_path = state_data_dir / f"y_{split}.npy"
    probabilities_path = transition_data_dir / "arrays" / f"expert_transition_probabilities_{split}.npy"
    labels = np.load(labels_path, mmap_mode="r")
    probabilities = np.load(probabilities_path, mmap_mode="r")
    if labels.shape != (probabilities.shape[0], NUM_EXPERTS) or labels.dtype != np.uint8:
        raise ValueError(f"{split}: label incompatibili con la feature di transizione.")
    if probabilities.shape[1] != NUM_EXPERTS or probabilities.dtype != np.float16:
        raise ValueError(f"{split}: probabilita' incompatibili: {probabilities_path}")
    if not np.all(labels.sum(axis=1) == TOP_K):
        raise ValueError(f"{split}: ogni label deve contenere esattamente {TOP_K} expert.")
    return labels, probabilities


def evaluate(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    # argpartition avoids sorting all 256 components.  Order within the six
    # selected indices is irrelevant for Recall@6.
    predictions = np.argpartition(probabilities, kth=NUM_EXPERTS - TOP_K, axis=1)[:, -TOP_K:]
    row_ids = np.arange(labels.shape[0])[:, None]
    hits_per_example = labels[row_ids, predictions].sum(axis=1, dtype=np.int16)
    recall = float(hits_per_example.sum() / (labels.shape[0] * TOP_K))
    histogram = np.bincount(hits_per_example, minlength=TOP_K + 1)
    return {
        "examples": int(labels.shape[0]),
        "recall_at_6": recall,
        "mean_target_experts_recovered": float(hits_per_example.mean()),
        "recovered_experts_histogram": {str(hits): int(count) for hits, count in enumerate(histogram)},
    }


def main() -> None:
    args = parse_args()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        raise ValueError("Specificare almeno uno split.")
    results: dict[str, object] = {
        "experiment": "expert_transition_probabilities_top6_baseline",
        "prediction_rule": "top-6 delle componenti di q_{t,l}, senza addestramento",
        "transition_feature_root": str(args.expert_transition_probabilities_data_dir),
        "splits": {},
    }
    for split in splits:
        labels, probabilities = load_split(
            args.state_data_dir, args.expert_transition_probabilities_data_dir, split
        )
        summary = evaluate(labels, probabilities)
        results["splits"][split] = summary
        print(f"{split}: Recall@6={summary['recall_at_6']:.6f}; "
              f"expert target recuperati in media={summary['mean_target_experts_recovered']:.3f}/6")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
