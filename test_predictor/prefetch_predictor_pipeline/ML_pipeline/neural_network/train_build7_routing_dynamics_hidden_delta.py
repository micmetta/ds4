#!/usr/bin/env python3
"""Screen the two validated Build 7 candidate families together.

The base Top-4 architecture is unchanged.  Hidden delta and routing dynamics
each have their own projection and additive expert-logit residual, preserving
their different calibration and making the combined result interpretable.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from ffn_utils import (
    RANDOM_STATE, build_fixed_nnz_state_two_hidden_gated_residual_two_candidate_ffn,
    calculate_recall_at_6, load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs, load_labels, load_quantized_hidden_inputs,
    require_new_output_directory, require_tensorflow, save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-features-dir", type=Path, required=True)
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--current-fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-delta-units", type=int, default=64)
    parser.add_argument("--routing-dynamics-units", type=int, default=128)
    parser.add_argument("--resume-weights", type=Path, default=None,
                        help="Checkpoint best.weights.h5 da cui riprendere; l'optimizer viene reinizializzato.")
    parser.add_argument("--initial-epoch", type=int, default=0)
    parser.add_argument("--resume-best-validation-recall", type=float, default=None)
    return parser.parse_args()


def array(path: Path, count: int, width: int, dtype: np.dtype) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Feature candidata non trovata: {path}")
    values = np.load(path, mmap_mode="r")
    if values.shape != (count, width) or values.dtype != dtype:
        raise ValueError(f"{path}: attesa {(count, width)}/{dtype}, trovata {values.shape}/{values.dtype}.")
    return values


def candidates(root: Path, split: str, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    delta_root = root / "01_hidden_delta"
    delta = array(delta_root / "arrays" / f"hidden_delta_int8_{split}.npy", count, 4096, np.int8)
    delta_aux = array(delta_root / "arrays" / f"hidden_delta_statistics_{split}.npy", count, 4, np.float16)
    routing = array(root / "02_routing_dynamics" / "arrays" / f"routing_recency_stability_{split}.npy", count, 261, np.float16)
    scale = float(json.loads((delta_root / "artifacts" / "materialization_summary.json").read_text(encoding="utf-8"))["delta_dequantization_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scala hidden delta non valida.")
    return delta, delta_aux, routing, scale


def dataset(indices, values, labels, current, previous, transition, delta, delta_aux, routing, batch_size, training, tf):
    examples, slots = indices.shape
    epoch = 0
    signature = {
        "state_feature_indices": tf.TensorSpec((None, slots), tf.int32),
        "state_feature_values": tf.TensorSpec((None, slots), tf.float32),
        "raw_hidden_current": tf.TensorSpec((None, 4096), tf.int8),
        "raw_hidden_previous": tf.TensorSpec((None, 4096), tf.int8),
        "expert_transition_probabilities": tf.TensorSpec((None, 256), tf.float16),
        "hidden_delta_features": tf.TensorSpec((None, 4096), tf.int8),
        "hidden_delta_aux_features": tf.TensorSpec((None, 4), tf.float16),
        "routing_dynamics_features": tf.TensorSpec((None, 261), tf.float16),
    }
    def batches():
        nonlocal epoch
        order = np.arange(examples)
        if training:
            np.random.default_rng(RANDOM_STATE + epoch).shuffle(order)
            epoch += 1
        for start in range(0, examples, batch_size):
            rows = order[start:min(start + batch_size, examples)] if training else slice(start, min(start + batch_size, examples))
            yield {
                "state_feature_indices": indices[rows].astype(np.int32, copy=False),
                "state_feature_values": values[rows].astype(np.float32, copy=False),
                "raw_hidden_current": current[rows], "raw_hidden_previous": previous[rows],
                "expert_transition_probabilities": transition[rows].astype(np.float16, copy=False),
                "hidden_delta_features": delta[rows],
                "hidden_delta_aux_features": delta_aux[rows].astype(np.float16, copy=False),
                "routing_dynamics_features": routing[rows].astype(np.float16, copy=False),
            }, labels[rows].astype(np.float32, copy=False)
    return tf.data.Dataset.from_generator(batches, output_signature=(signature, tf.TensorSpec((None, 256), tf.float32))).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    if (args.epochs < 1 or args.initial_epoch < 0 or args.initial_epoch >= args.epochs
            or args.early_stopping_patience < 0 or args.batch_size < 1):
        raise ValueError("Iperparametri non validi.")
    if args.resume_weights is None and (args.initial_epoch or args.resume_best_validation_recall is not None):
        raise ValueError("--initial-epoch e --resume-best-validation-recall richiedono --resume-weights.")
    if args.resume_weights is not None:
        if not args.resume_weights.is_file():
            raise FileNotFoundError(f"Checkpoint di ripresa non trovato: {args.resume_weights}")
        if args.resume_best_validation_recall is None or not 0.0 <= args.resume_best_validation_recall <= 1.0:
            raise ValueError("Con --resume-weights specifica una --resume-best-validation-recall valida.")
    require_new_output_directory(args.output_dir)
    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    print("1) Carico Build 7 Top-4, hidden_delta e routing_dynamics...")
    y_train, y_validation = load_labels(args.state_data_dir, "train"), load_labels(args.state_data_dir, "validation")
    ti, tv, state_meta = load_fixed_nnz_state_inputs(args.current_fixed_nnz_state_data_dir, "train", y_train.shape[0])
    vi, vv, validation_meta = load_fixed_nnz_state_inputs(args.current_fixed_nnz_state_data_dir, "validation", y_validation.shape[0])
    if state_meta != validation_meta:
        raise ValueError("Metadata state train/validation diverse.")
    current_train, current_scale = load_quantized_hidden_inputs(args.current_quantized_hidden_data_dir, "train", y_train.shape[0])
    current_validation, validation_current_scale = load_quantized_hidden_inputs(args.current_quantized_hidden_data_dir, "validation", y_validation.shape[0])
    previous_train, previous_scale = load_quantized_hidden_inputs(args.previous_quantized_hidden_data_dir, "train", y_train.shape[0])
    previous_validation, validation_previous_scale = load_quantized_hidden_inputs(args.previous_quantized_hidden_data_dir, "validation", y_validation.shape[0])
    if (current_scale, previous_scale) != (validation_current_scale, validation_previous_scale):
        raise ValueError("Scale raw hidden train/validation diverse.")
    transition_train = load_expert_transition_probabilities_inputs(args.expert_transition_probabilities_data_dir, "train", y_train.shape[0])
    transition_validation = load_expert_transition_probabilities_inputs(args.expert_transition_probabilities_data_dir, "validation", y_validation.shape[0])
    delta_train, delta_aux_train, routing_train, delta_scale = candidates(args.candidate_features_dir, "train", y_train.shape[0])
    delta_validation, delta_aux_validation, routing_validation, validation_delta_scale = candidates(args.candidate_features_dir, "validation", y_validation.shape[0])
    if delta_scale != validation_delta_scale:
        raise ValueError("Scale hidden delta train/validation diversa.")
    print(f"   hidden_delta={delta_train.shape}+{delta_aux_train.shape}; routing_dynamics={routing_train.shape}")
    model = build_fixed_nnz_state_two_hidden_gated_residual_two_candidate_ffn(
        state_meta["feature_count"], state_meta["input_slots"], 2048, 64, 64,
        current_scale, previous_scale, 2048, 0.3, 1.5e-4, 3e-6,
        4096, args.hidden_delta_units, "int8", delta_scale, 4,
        261, args.routing_dynamics_units, "float16", tf,
    )
    if args.resume_weights is not None:
        model.load_weights(args.resume_weights)
        print(f"   Riprendo da {args.resume_weights} dopo {args.initial_epoch} epoche; "
              f"best val Recall@6 gia' nota={args.resume_best_validation_recall:.6f}.")
    train = dataset(ti, tv, y_train, current_train, previous_train, transition_train, delta_train, delta_aux_train, routing_train, args.batch_size, True, tf).repeat()
    validation = dataset(vi, vv, y_validation, current_validation, previous_validation, transition_validation, delta_validation, delta_aux_validation, routing_validation, args.batch_size, False, tf).repeat()
    train_steps, validation_steps = math.ceil(y_train.shape[0] / args.batch_size), math.ceil(y_validation.shape[0] / args.batch_size)
    models, reports = args.output_dir / "models", args.output_dir / "reports"
    models.mkdir(parents=True); reports.mkdir(parents=True)
    best = models / "best.weights.h5"
    checkpoint_kwargs = {}
    if args.resume_weights is not None:
        shutil.copy2(args.resume_weights, best)
        checkpoint_kwargs["initial_value_threshold"] = args.resume_best_validation_recall
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best), monitor="val_recall_at_6", mode="max", save_best_only=True,
                                           save_weights_only=True, **checkpoint_kwargs),
        tf.keras.callbacks.EarlyStopping(monitor="val_recall_at_6", mode="max", patience=args.early_stopping_patience, restore_best_weights=True),
    ]
    print(f"2) Addestro da epoca {args.initial_epoch + 1} fino a {args.epochs}, patience={args.early_stopping_patience}...")
    started = time.perf_counter()
    history = model.fit(train, steps_per_epoch=train_steps, validation_data=validation, validation_steps=validation_steps,
                        initial_epoch=args.initial_epoch, epochs=args.epochs, callbacks=callbacks, verbose=2)
    elapsed = time.perf_counter() - started
    model.load_weights(best)
    recall = calculate_recall_at_6(model.predict(validation, steps=validation_steps, verbose=0), y_validation)
    model_path = models / "ffn.keras"; model.save(model_path)
    save_json(reports / "training_report.json", {
        "experiment": "build7_top4_hidden_delta_routing_dynamics", "best_validation_recall_at_6": recall,
        "elapsed_seconds": elapsed, "hidden_delta_units": args.hidden_delta_units,
        "routing_dynamics_units": args.routing_dynamics_units, "model_path": str(model_path), "best_weights_path": str(best),
        "initial_epoch": args.initial_epoch,
        "resumed_from_weights": None if args.resume_weights is None else str(args.resume_weights),
        "resume_best_validation_recall": args.resume_best_validation_recall,
        "optimizer_state_restored": False,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    })
    print(f"Completato hidden_delta+routing_dynamics: validation Recall@6 {recall:.6f}")


if __name__ == "__main__":
    main()
