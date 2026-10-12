#!/usr/bin/env python3
"""Train one causal priority-feature family on top of the fixed Build 7 Top-4 path."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from ffn_utils import (
    RANDOM_STATE, build_fixed_nnz_state_two_hidden_gated_residual_candidate_ffn,
    calculate_recall_at_6, load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs, load_labels, load_quantized_hidden_inputs,
    require_new_output_directory, require_tensorflow, save_json,
)


FAMILIES = {
    "hidden_delta": {"array": "hidden_delta_int8", "directory": "01_hidden_delta", "dtype": "int8", "dim": 4096,
                     "aux_array": "hidden_delta_statistics", "aux_dim": 4, "units": 64},
    "routing_dynamics": {"array": "routing_recency_stability", "directory": "02_routing_dynamics", "dtype": "float16", "dim": 261,
                         "aux_array": None, "aux_dim": 0, "units": 128},
    "layer_expert_ema": {"array": "layer_expert_ema", "directory": "03_layer_expert_ema", "dtype": "float16", "dim": 256,
                         "aux_array": None, "aux_dim": 0, "units": 128},
    "router_confidence": {"array": "router_confidence", "directory": "04_router_confidence", "dtype": "float16", "dim": 33,
                          "aux_array": None, "aux_dim": 0, "units": 64},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-family", choices=tuple(FAMILIES), required=True)
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
    parser.add_argument("--candidate-units", type=int, default=None)
    parser.add_argument("--resume-weights", type=Path, default=None,
                        help="Pesi best.weights.h5 da cui riprendere; l'optimizer viene reinizializzato.")
    parser.add_argument("--initial-epoch", type=int, default=0,
                        help="Numero di epoche gia' completate, usato solo per la numerazione della ripresa.")
    parser.add_argument("--resume-best-validation-recall", type=float, default=None,
                        help="Recall@6 del checkpoint di ripresa: impedisce di sostituirlo con epoche peggiori.")
    return parser.parse_args()


def candidate_paths(root: Path, family: dict[str, object], split: str) -> tuple[Path, Path | None]:
    arrays = root / str(family["directory"]) / "arrays"
    primary = arrays / f"{family['array']}_{split}.npy"
    auxiliary = None if family["aux_array"] is None else arrays / f"{family['aux_array']}_{split}.npy"
    return primary, auxiliary


def load_candidate(root: Path, family: dict[str, object], split: str, count: int):
    primary_path, auxiliary_path = candidate_paths(root, family, split)
    if not primary_path.is_file():
        raise FileNotFoundError(f"Candidate feature non trovata: {primary_path}")
    primary = np.load(primary_path, mmap_mode="r")
    if primary.shape != (count, int(family["dim"])) or str(primary.dtype) != str(family["dtype"]):
        raise ValueError(f"{primary_path}: shape/dtype incompatibile: {primary.shape}/{primary.dtype}")
    auxiliary = None
    if auxiliary_path is not None:
        auxiliary = np.load(auxiliary_path, mmap_mode="r")
        if auxiliary.shape != (count, int(family["aux_dim"])) or auxiliary.dtype != np.float16:
            raise ValueError(f"{auxiliary_path}: shape/dtype incompatibile: {auxiliary.shape}/{auxiliary.dtype}")
    return primary, auxiliary


def delta_scale(root: Path) -> float:
    path = root / "01_hidden_delta" / "artifacts" / "materialization_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    scale = float(payload["delta_dequantization_scale"])
    if scale <= 0:
        raise ValueError("Scala hidden delta non valida.")
    return scale


def make_dataset(indices, values, labels, current_hidden, previous_hidden, transition, candidate, candidate_aux, batch_size, training, tf):
    examples, slots = indices.shape
    epoch = 0
    candidate_dim = candidate.shape[1]
    signature = {
        "state_feature_indices": tf.TensorSpec((None, slots), tf.int32),
        "state_feature_values": tf.TensorSpec((None, slots), tf.float32),
        "raw_hidden_current": tf.TensorSpec((None, 4096), tf.int8),
        "raw_hidden_previous": tf.TensorSpec((None, 4096), tf.int8),
        "expert_transition_probabilities": tf.TensorSpec((None, 256), tf.float16),
        "candidate_features": tf.TensorSpec((None, candidate_dim), tf.as_dtype(candidate.dtype)),
    }
    if candidate_aux is not None:
        signature["candidate_aux_features"] = tf.TensorSpec((None, candidate_aux.shape[1]), tf.float16)

    def batches():
        nonlocal epoch
        order = np.arange(examples)
        if training:
            np.random.default_rng(RANDOM_STATE + epoch).shuffle(order)
            epoch += 1
        for start in range(0, examples, batch_size):
            rows = order[start:min(start + batch_size, examples)] if training else slice(start, min(start + batch_size, examples))
            payload = {
                "state_feature_indices": indices[rows].astype(np.int32, copy=False),
                "state_feature_values": values[rows].astype(np.float32, copy=False),
                "raw_hidden_current": current_hidden[rows],
                "raw_hidden_previous": previous_hidden[rows],
                "expert_transition_probabilities": transition[rows].astype(np.float16, copy=False),
                "candidate_features": candidate[rows],
            }
            if candidate_aux is not None:
                payload["candidate_aux_features"] = candidate_aux[rows].astype(np.float16, copy=False)
            yield payload, labels[rows].astype(np.float32, copy=False)
    return tf.data.Dataset.from_generator(
        batches, output_signature=(signature, tf.TensorSpec((None, 256), tf.float32))
    ).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    if (args.epochs < 1 or args.initial_epoch < 0 or args.initial_epoch >= args.epochs
            or args.early_stopping_patience < 0 or args.batch_size < 1):
        raise ValueError("Iperparametri non validi.")
    if args.resume_weights is None and (args.initial_epoch != 0 or args.resume_best_validation_recall is not None):
        raise ValueError("--initial-epoch e --resume-best-validation-recall richiedono --resume-weights.")
    if args.resume_weights is not None:
        if not args.resume_weights.is_file():
            raise FileNotFoundError(f"Checkpoint di ripresa non trovato: {args.resume_weights}")
        if args.resume_best_validation_recall is None:
            raise ValueError("Con --resume-weights specifica anche --resume-best-validation-recall.")
        if not 0.0 <= args.resume_best_validation_recall <= 1.0:
            raise ValueError("--resume-best-validation-recall non valida.")
    require_new_output_directory(args.output_dir)
    family = FAMILIES[args.candidate_family]
    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print(f"1) Carico Build 7 Top-4 e candidate family {args.candidate_family}...")
    y_train, y_validation = load_labels(args.state_data_dir, "train"), load_labels(args.state_data_dir, "validation")
    ti, tv, metadata = load_fixed_nnz_state_inputs(args.current_fixed_nnz_state_data_dir, "train", y_train.shape[0])
    vi, vv, validation_metadata = load_fixed_nnz_state_inputs(args.current_fixed_nnz_state_data_dir, "validation", y_validation.shape[0])
    if metadata != validation_metadata:
        raise ValueError("Metadata state train/validation diverse.")
    train_current, current_scale = load_quantized_hidden_inputs(args.current_quantized_hidden_data_dir, "train", y_train.shape[0])
    validation_current, validation_current_scale = load_quantized_hidden_inputs(args.current_quantized_hidden_data_dir, "validation", y_validation.shape[0])
    train_previous, previous_scale = load_quantized_hidden_inputs(args.previous_quantized_hidden_data_dir, "train", y_train.shape[0])
    validation_previous, validation_previous_scale = load_quantized_hidden_inputs(args.previous_quantized_hidden_data_dir, "validation", y_validation.shape[0])
    if (current_scale, previous_scale) != (validation_current_scale, validation_previous_scale):
        raise ValueError("Scale raw hidden train/validation diverse.")
    train_transition = load_expert_transition_probabilities_inputs(args.expert_transition_probabilities_data_dir, "train", y_train.shape[0])
    validation_transition = load_expert_transition_probabilities_inputs(args.expert_transition_probabilities_data_dir, "validation", y_validation.shape[0])
    train_candidate, train_aux = load_candidate(args.candidate_features_dir, family, "train", y_train.shape[0])
    validation_candidate, validation_aux = load_candidate(args.candidate_features_dir, family, "validation", y_validation.shape[0])
    units = args.candidate_units or int(family["units"])
    scale = delta_scale(args.candidate_features_dir) if family["dtype"] == "int8" else None

    print(f"   candidate={train_candidate.shape}; aux={None if train_aux is None else train_aux.shape}; units={units}")
    model = build_fixed_nnz_state_two_hidden_gated_residual_candidate_ffn(
        metadata["feature_count"], metadata["input_slots"], 2048, 64, 64,
        current_scale, previous_scale, 2048, 0.3, 1.5e-4, 3e-6,
        int(family["dim"]), units, str(family["dtype"]), tf,
        candidate_scale=scale, candidate_aux_dimension=int(family["aux_dim"]),
    )
    if args.resume_weights is not None:
        model.load_weights(args.resume_weights)
        print(f"   Riprendo da {args.resume_weights} dopo {args.initial_epoch} epoche; "
              f"best val Recall@6 gia' nota={args.resume_best_validation_recall:.6f}.")
    train_dataset = make_dataset(ti, tv, y_train, train_current, train_previous, train_transition, train_candidate, train_aux, args.batch_size, True, tf).repeat()
    validation_dataset = make_dataset(vi, vv, y_validation, validation_current, validation_previous, validation_transition, validation_candidate, validation_aux, args.batch_size, False, tf).repeat()
    train_steps = math.ceil(y_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / args.batch_size)
    models, reports = args.output_dir / "models", args.output_dir / "reports"
    models.mkdir(parents=True); reports.mkdir(parents=True)
    best = models / "best.weights.h5"
    checkpoint_kwargs = {}
    if args.resume_weights is not None:
        # Il nuovo callback parte con best=+/-inf; copiamo il checkpoint precedente e
        # imponiamo la sua metrica come soglia, cosi' non viene degradato dalla ripresa.
        shutil.copy2(args.resume_weights, best)
        checkpoint_kwargs["initial_value_threshold"] = args.resume_best_validation_recall
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best), monitor="val_recall_at_6", mode="max", save_best_only=True,
                                           save_weights_only=True, **checkpoint_kwargs),
        tf.keras.callbacks.EarlyStopping(monitor="val_recall_at_6", mode="max", patience=args.early_stopping_patience, restore_best_weights=True),
    ]
    print(f"2) Addestro da epoca {args.initial_epoch + 1} fino a {args.epochs}, "
          f"patience={args.early_stopping_patience}...")
    started = time.perf_counter()
    history = model.fit(train_dataset, steps_per_epoch=train_steps, validation_data=validation_dataset,
                        validation_steps=validation_steps, initial_epoch=args.initial_epoch,
                        epochs=args.epochs, callbacks=callbacks, verbose=2)
    elapsed = time.perf_counter() - started
    model.load_weights(best)
    scores = model.predict(validation_dataset, steps=validation_steps, verbose=0)
    recall = calculate_recall_at_6(scores, y_validation)
    model_path = models / "ffn.keras"; model.save(model_path)
    save_json(reports / "training_report.json", {
        "experiment": "build7_top4_single_priority_feature_candidate", "candidate_family": args.candidate_family,
        "best_validation_recall_at_6": recall, "elapsed_seconds": elapsed, "candidate_units": units,
        "initial_epoch": args.initial_epoch,
        "resumed_from_weights": None if args.resume_weights is None else str(args.resume_weights),
        "resume_best_validation_recall": args.resume_best_validation_recall,
        "optimizer_state_restored": False,
        "candidate_shape": list(train_candidate.shape), "candidate_aux_shape": None if train_aux is None else list(train_aux.shape),
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "model_path": str(model_path), "best_weights_path": str(best),
    })
    print(f"Completato {args.candidate_family}: validation Recall@6 {recall:.6f}")


if __name__ == "__main__":
    main()
