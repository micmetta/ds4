#!/usr/bin/env python3
"""Measure zero/permutation importance of one trained Build 7 priority-feature branch."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path
from zlib import crc32

import numpy as np

from ffn_utils import (
    RANDOM_STATE, build_fixed_nnz_state_two_hidden_gated_residual_candidate_ffn,
    calculate_recall_at_6, load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs, load_labels, load_quantized_hidden_inputs,
    require_tensorflow, save_json,
)
from train_build7_priority_feature_candidate import FAMILIES, delta_scale, load_candidate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--candidate-family", choices=tuple(FAMILIES), required=True)
    p.add_argument("--candidate-features-dir", type=Path, required=True)
    p.add_argument("--state-data-dir", type=Path, required=True)
    p.add_argument("--current-fixed-nnz-state-data-dir", type=Path, required=True)
    p.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    p.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    p.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    p.add_argument("--output-report", type=Path, required=True)
    p.add_argument("--splits", default="validation")
    p.add_argument("--modes", default="zero,permutation")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--candidate-units", type=int, default=None)
    return p.parse_args()


def csv(value: str, allowed: tuple[str, ...]) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or set(result) - set(allowed):
        raise ValueError(f"Valore non valido: {value}; ammessi: {', '.join(allowed)}")
    return result


def load_keras_weights(path: Path, model) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        weights = Path(temporary) / "model.weights.h5"
        with zipfile.ZipFile(path) as archive, archive.open("model.weights.h5") as source, weights.open("wb") as destination:
            destination.write(source.read())
        model.load_weights(weights)


def load_split(args: argparse.Namespace, family: dict[str, object], split: str):
    labels = load_labels(args.state_data_dir, split)
    indices, values, metadata = load_fixed_nnz_state_inputs(args.current_fixed_nnz_state_data_dir, split, labels.shape[0])
    current, current_scale = load_quantized_hidden_inputs(args.current_quantized_hidden_data_dir, split, labels.shape[0])
    previous, previous_scale = load_quantized_hidden_inputs(args.previous_quantized_hidden_data_dir, split, labels.shape[0])
    candidate, auxiliary = load_candidate(args.candidate_features_dir, family, split, labels.shape[0])
    return {"labels": labels, "indices": indices, "values": values, "metadata": metadata,
            "current": current, "previous": previous, "transition": load_expert_transition_probabilities_inputs(args.expert_transition_probabilities_data_dir, split, labels.shape[0]),
            "candidate": candidate, "auxiliary": auxiliary, "current_scale": current_scale, "previous_scale": previous_scale}


def dataset(data, batch_size, tf, mode: str | None):
    count, slots = data["indices"].shape
    permutation = np.random.default_rng(RANDOM_STATE + crc32((mode or "reference").encode())).permutation(count) if mode == "permutation" else None
    candidate_dim = data["candidate"].shape[1]
    signature = {"state_feature_indices": tf.TensorSpec((None, slots), tf.int32), "state_feature_values": tf.TensorSpec((None, slots), tf.float32),
                 "raw_hidden_current": tf.TensorSpec((None, 4096), tf.int8), "raw_hidden_previous": tf.TensorSpec((None, 4096), tf.int8),
                 "expert_transition_probabilities": tf.TensorSpec((None, 256), tf.float16),
                 "candidate_features": tf.TensorSpec((None, candidate_dim), tf.as_dtype(data["candidate"].dtype))}
    if data["auxiliary"] is not None:
        signature["candidate_aux_features"] = tf.TensorSpec((None, data["auxiliary"].shape[1]), tf.float16)
    def batches():
        for begin in range(0, count, batch_size):
            rows = slice(begin, min(begin + batch_size, count))
            candidate_rows = permutation[rows] if permutation is not None else rows
            candidate = data["candidate"][candidate_rows]
            auxiliary = None if data["auxiliary"] is None else data["auxiliary"][candidate_rows]
            if mode == "zero":
                candidate = np.zeros(candidate.shape, dtype=candidate.dtype)
                if auxiliary is not None:
                    auxiliary = np.zeros(auxiliary.shape, dtype=auxiliary.dtype)
            payload = {"state_feature_indices": data["indices"][rows].astype(np.int32, copy=False), "state_feature_values": data["values"][rows].astype(np.float32, copy=False),
                       "raw_hidden_current": data["current"][rows], "raw_hidden_previous": data["previous"][rows],
                       "expert_transition_probabilities": data["transition"][rows].astype(np.float16, copy=False), "candidate_features": candidate}
            if auxiliary is not None:
                payload["candidate_aux_features"] = auxiliary.astype(np.float16, copy=False)
            yield payload, data["labels"][rows].astype(np.float32, copy=False)
    return tf.data.Dataset.from_generator(batches, output_signature=(signature, tf.TensorSpec((None, 256), tf.float32))).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    if args.output_report.exists():
        raise FileExistsError(f"Report gia' esistente: {args.output_report}")
    splits, modes = csv(args.splits, ("train", "validation", "test")), csv(args.modes, ("zero", "permutation"))
    family = FAMILIES[args.candidate_family]
    tf = require_tensorflow(); tf.keras.mixed_precision.set_global_policy("float32")
    data = {split: load_split(args, family, split) for split in splits}
    reference = data[splits[0]]
    units = args.candidate_units
    if units is None:
        report = args.checkpoint.parent.parent / "reports" / "training_report.json"
        units = int(json.loads(report.read_text(encoding="utf-8"))["candidate_units"])
    scale = delta_scale(args.candidate_features_dir) if family["dtype"] == "int8" else None
    model = build_fixed_nnz_state_two_hidden_gated_residual_candidate_ffn(
        reference["metadata"]["feature_count"], reference["metadata"]["input_slots"], 2048, 64, 64,
        reference["current_scale"], reference["previous_scale"], 2048, 0.3, 1.5e-4, 3e-6,
        int(family["dim"]), units, str(family["dtype"]), tf, candidate_scale=scale,
        candidate_aux_dimension=int(family["aux_dim"]),
    )
    load_keras_weights(args.checkpoint, model)
    result = {"candidate_family": args.candidate_family, "checkpoint": str(args.checkpoint), "candidate_units": units, "splits": {}}
    for split in splits:
        reference_recall = calculate_recall_at_6(model.predict(dataset(data[split], args.batch_size, tf, None), steps=math.ceil(data[split]["labels"].shape[0] / args.batch_size), verbose=0), data[split]["labels"])
        entry = {"reference_recall_at_6": reference_recall, "ablations": {}}
        for mode in modes:
            recall = calculate_recall_at_6(model.predict(dataset(data[split], args.batch_size, tf, mode), steps=math.ceil(data[split]["labels"].shape[0] / args.batch_size), verbose=0), data[split]["labels"])
            entry["ablations"][mode] = {"recall_at_6": recall, "drop_from_reference": reference_recall - recall}
        result["splits"][split] = entry
        print(f"{split}: riferimento={reference_recall:.6f}; " + "; ".join(f"{m}={entry['ablations'][m]['recall_at_6']:.6f}" for m in modes))
    save_json(args.output_report, result)
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
