#!/usr/bin/env python3
"""Successive-halving Top-4 Build 7: 36 -> 12 -> 6 candidates only.

The script intentionally stops after phase A3.  It does not use the test set
and it does not run a final 1,000-epoch retraining: that choice is made by the
user after inspecting the six A3 finalists.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ffn_utils import (
    RANDOM_STATE,
    build_fixed_nnz_state_two_hidden_gated_residual_ffn,
    load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_quantized_hidden_inputs,
    require_tensorflow,
)


STATE_UNITS = (1024, 2048, 4096)
CURRENT_HIDDEN_UNITS = (64, 256, 1024)
PREVIOUS_HIDDEN_UNITS = (64, 256)
FUSION_UNITS = (1024, 2048)
PHASES = (
    ("A1", 5, 36, 12, None),
    ("A2", 20, 12, 6, None),
    ("A3", 60, 6, None, 5),
)
EXPERIMENT_NAME = "build7_top4_successive_halving_a1_a3"
SEARCH_LABEL = "A"
EXPECTED_STATE_METADATA = {
    "feature_count": 4411,
    "nonzero_features_per_example": 44,
    "input_slots": 45,
}


def write_json_atomically(path: Path, content: Any) -> None:
    """Write JSON without leaving a partial state after an interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuning Top-4 Build 7 A1-A3: 36 -> 12 -> 6, senza test e senza fase finale automatica."
    )
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--current-fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Riprende dalla prima fase incompleta.")
    parser.add_argument("--l2-regularization", type=float, default=3e-6)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout fisso dopo state e fusione; i raw hidden restano senza dropout.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--shuffle-block-size", type=int, default=65536)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (args.batch_size < 1 or args.shuffle_block_size < args.batch_size
            or args.l2_regularization < 0 or args.learning_rate <= 0
            or not 0 <= args.dropout < 1):
        raise ValueError("Iperparametri del tuning non validi.")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output gia' esistente: {args.output_dir}. Usa --resume o una nuova cartella.")
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError(f"Nessun tuning da riprendere in: {args.output_dir}")


def immutable_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "data": {
            "state_data_dir": str(args.state_data_dir),
            "current_fixed_nnz_state_data_dir": str(args.current_fixed_nnz_state_data_dir),
            "current_quantized_hidden_data_dir": str(args.current_quantized_hidden_data_dir),
            "previous_quantized_hidden_data_dir": str(args.previous_quantized_hidden_data_dir),
            "expert_transition_probabilities_data_dir": str(args.expert_transition_probabilities_data_dir),
        },
        "search_space": {
            "state_hidden_units": list(STATE_UNITS),
            "current_hidden_projection_dimension": list(CURRENT_HIDDEN_UNITS),
            "previous_hidden_projection_dimension": list(PREVIOUS_HIDDEN_UNITS),
            "fusion_hidden_units": list(FUSION_UNITS),
            "raw_hidden_branch_dropout": 0.0,
        },
        "phases": [
            {"name": name, "target_epochs": epochs, "candidates": candidates,
             "promote": promote, "early_stopping_patience": patience}
            for name, epochs, candidates, promote, patience in PHASES
        ],
        "fixed_hyperparameters": {
            "loss": "binary_cross_entropy",
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
            "state_and_fusion_dropout": args.dropout,
            "raw_hidden_branch_dropout": 0.0,
            "batch_size": args.batch_size,
            "shuffle_block_size": args.shuffle_block_size,
            "tensorflow_policy": "float32",
        },
        "selection_metric": "best validation Recall@6 observed across all completed phases",
        "random_state": RANDOM_STATE,
    }


def all_candidates() -> list[dict[str, Any]]:
    candidates = []
    for number, values in enumerate(
        itertools.product(STATE_UNITS, CURRENT_HIDDEN_UNITS, PREVIOUS_HIDDEN_UNITS, FUSION_UNITS), start=1
    ):
        state_units, current_units, previous_units, fusion_units = values
        candidates.append({
            "candidate_id": f"candidate_{number:02d}",
            "hyperparameters": {
                "state_hidden_units": state_units,
                "current_hidden_projection_dimension": current_units,
                "previous_hidden_projection_dimension": previous_units,
                "fusion_hidden_units": fusion_units,
                "raw_hidden_branch_dropout": 0.0,
            },
            "phases": {},
        })
    if len(candidates) != 36:
        raise AssertionError("Lo spazio Top-4 A deve contenere esattamente 36 candidate.")
    return candidates


def checkpoint_path(root: Path, candidate_id: str, phase_name: str) -> Path:
    return root / "candidate_checkpoints" / candidate_id / f"{phase_name}.model_and_adam.npz"


def save_model_and_adam(path: Path, model) -> None:
    """Save tensors and Adam slots so promoted candidates resume exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    payload = {
        **{f"model_{index}": value for index, value in enumerate(model.get_weights())},
        **{f"adam_{index}": variable.numpy() for index, variable in enumerate(model.optimizer.variables)},
    }
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def load_model_and_adam(path: Path, model) -> None:
    """Restore model and optimizer from a stable promotion checkpoint."""
    with np.load(path) as archive:
        model_keys = sorted((key for key in archive.files if key.startswith("model_")),
                            key=lambda key: int(key.removeprefix("model_")))
        adam_keys = sorted((key for key in archive.files if key.startswith("adam_")),
                           key=lambda key: int(key.removeprefix("adam_")))
        model.set_weights([archive[key] for key in model_keys])
        saved_adam = [archive[key] for key in adam_keys]
    model.optimizer.build(model.trainable_variables)
    variables = list(model.optimizer.variables)
    if len(saved_adam) != len(variables):
        raise ValueError("Checkpoint Adam incompatibile con la candidata promossa.")
    for variable, value in zip(variables, saved_adam, strict=True):
        if tuple(variable.shape) != tuple(value.shape):
            raise ValueError("Checkpoint Adam incompatibile: shape diversa.")
        variable.assign(value)


def make_dataset(indices, values, labels, current_hidden, previous_hidden, transition,
                 batch_size: int, training: bool, initial_epoch: int, tf):
    """Create a deterministic Top-4 dataset; candidates see the same epoch order."""
    examples, slots = indices.shape
    epoch_number = initial_epoch

    def batches():
        nonlocal epoch_number
        order = np.arange(examples)
        if training:
            np.random.default_rng(RANDOM_STATE + epoch_number).shuffle(order)
            epoch_number += 1
        for start in range(0, examples, batch_size):
            stop = min(start + batch_size, examples)
            rows = order[start:stop] if training else slice(start, stop)
            yield {
                "state_feature_indices": indices[rows].astype(np.int32, copy=False),
                "state_feature_values": values[rows].astype(np.float32, copy=False),
                "raw_hidden_current": current_hidden[rows],
                "raw_hidden_previous": previous_hidden[rows],
                "expert_transition_probabilities": transition[rows].astype(np.float16, copy=False),
            }, labels[rows].astype(np.float32, copy=False)

    signature = ({
        "state_feature_indices": tf.TensorSpec((None, slots), tf.int32),
        "state_feature_values": tf.TensorSpec((None, slots), tf.float32),
        "raw_hidden_current": tf.TensorSpec((None, 4096), tf.int8),
        "raw_hidden_previous": tf.TensorSpec((None, 4096), tf.int8),
        "expert_transition_probabilities": tf.TensorSpec((None, 256), tf.float16),
    }, tf.TensorSpec((None, 256), tf.float32))
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    validate_args(args)
    reports_dir = args.output_dir / "reports"
    configuration = immutable_configuration(args)
    configuration_path = reports_dir / "search_configuration.json"
    state_path = reports_dir / "tuning_state.json"
    if args.resume:
        if not configuration_path.is_file() or not state_path.is_file():
            raise FileNotFoundError("Mancano configurazione o stato: impossibile riprendere in sicurezza.")
        if json.loads(configuration_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("La configurazione richiesta differisce dal tuning gia' iniziato.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        args.output_dir.mkdir(parents=True)
        state = {"experiment": configuration["experiment"], "candidates": all_candidates(),
                 "completed_phases": [], "checkpoint_cleanup": []}
        write_json_atomically(configuration_path, configuration)
        write_json_atomically(state_path, state)

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    print("1) Carico Build 7 Top-4: state l, raw hidden l/l-1 e q 256D...")
    y_train = load_labels(args.state_data_dir, "train")
    y_validation = load_labels(args.state_data_dir, "validation")
    train_indices, train_values, metadata = load_fixed_nnz_state_inputs(
        args.current_fixed_nnz_state_data_dir, "train", y_train.shape[0]
    )
    validation_indices, validation_values, validation_metadata = load_fixed_nnz_state_inputs(
        args.current_fixed_nnz_state_data_dir, "validation", y_validation.shape[0]
    )
    if metadata != EXPECTED_STATE_METADATA or validation_metadata != metadata:
        raise ValueError("Il tuning Top-4 richiede lo state corrente 4411D con 45 slot.")
    train_current, current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, "train", y_train.shape[0]
    )
    validation_current, validation_current_scale = load_quantized_hidden_inputs(
        args.current_quantized_hidden_data_dir, "validation", y_validation.shape[0]
    )
    train_previous, previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, "train", y_train.shape[0]
    )
    validation_previous, validation_previous_scale = load_quantized_hidden_inputs(
        args.previous_quantized_hidden_data_dir, "validation", y_validation.shape[0]
    )
    if current_scale != validation_current_scale or previous_scale != validation_previous_scale:
        raise ValueError("Scale int8 diverse tra train e validation.")
    train_transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, "train", y_train.shape[0]
    )
    validation_transition = load_expert_transition_probabilities_inputs(
        args.expert_transition_probabilities_data_dir, "validation", y_validation.shape[0]
    )
    train_steps = math.ceil(y_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / args.batch_size)
    print(f"   train: state={train_indices.shape}; hidden={train_current.shape}/{train_previous.shape}; q={train_transition.shape}")

    def save_state() -> None:
        write_json_atomically(state_path, state)

    def best_score(candidate: dict[str, Any], phase_name: str) -> float:
        return float(candidate["phases"][phase_name]["best_validation_recall_at_6"])

    def remove_checkpoint(candidate_id: str, phase_name: str, reason: str) -> None:
        path = checkpoint_path(args.output_dir, candidate_id, phase_name)
        if path.is_file():
            path.unlink()
            state["checkpoint_cleanup"].append({"candidate_id": candidate_id, "phase": phase_name, "reason": reason})
            try:
                path.parent.rmdir()
            except OSError:
                pass

    class BestCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, path: Path, initial_best: float) -> None:
            super().__init__()
            self.path, self.best = path, initial_best

        def on_epoch_end(self, epoch, logs=None) -> None:
            score = float((logs or {}).get("val_recall_at_6", float("-inf")))
            if score > self.best:
                self.best = score
                save_model_and_adam(self.path, self.model)

    def train_phase(candidate: dict[str, Any], phase_name: str, target_epochs: int,
                    initial_epoch: int, patience: int | None) -> None:
        hyperparameters = candidate["hyperparameters"]
        candidate_id = candidate["candidate_id"]
        target_checkpoint = checkpoint_path(args.output_dir, candidate_id, phase_name)
        target_checkpoint.unlink(missing_ok=True)
        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.utils.set_random_seed(RANDOM_STATE)
        model = build_fixed_nnz_state_two_hidden_gated_residual_ffn(
            metadata["feature_count"], metadata["input_slots"], hyperparameters["state_hidden_units"],
            hyperparameters["current_hidden_projection_dimension"],
            hyperparameters["previous_hidden_projection_dimension"], current_scale, previous_scale,
            hyperparameters["fusion_hidden_units"], args.dropout, args.learning_rate,
            args.l2_regularization, tf,
        )
        previous_best = float("-inf")
        if initial_epoch:
            phase_index = [phase[0] for phase in PHASES].index(phase_name)
            previous_phase = PHASES[phase_index - 1][0]
            previous_record = candidate["phases"][previous_phase]
            load_model_and_adam(Path(previous_record["checkpoint_path"]), model)
            previous_best = float(previous_record["best_validation_recall_at_6"])
            save_model_and_adam(target_checkpoint, model)
        train_dataset = make_dataset(train_indices, train_values, y_train, train_current, train_previous,
                                     train_transition, args.batch_size, True, initial_epoch, tf).repeat()
        validation_dataset = make_dataset(validation_indices, validation_values, y_validation,
                                          validation_current, validation_previous, validation_transition,
                                          args.batch_size, False, 0, tf).repeat()
        callbacks: list[Any] = [BestCheckpoint(target_checkpoint, previous_best)]
        if patience is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_recall_at_6", mode="max", patience=patience, restore_best_weights=False
            ))
        started = time.perf_counter()
        try:
            history = model.fit(
                train_dataset, steps_per_epoch=train_steps, validation_data=validation_dataset,
                validation_steps=validation_steps, initial_epoch=initial_epoch, epochs=target_epochs,
                callbacks=callbacks, shuffle=False, verbose=2,
            )
            phase_scores = [float(value) for value in history.history.get("val_recall_at_6", [])]
            observed_best = max(phase_scores, default=float("-inf"))
            if not target_checkpoint.is_file():
                save_model_and_adam(target_checkpoint, model)
            candidate["phases"][phase_name] = {
                "status": "completed", "target_epochs": target_epochs, "initial_epoch": initial_epoch,
                "epochs_executed_in_phase": len(phase_scores),
                "best_validation_recall_at_6": max(previous_best, observed_best),
                "best_validation_recall_at_6_in_phase": observed_best,
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint_path": str(target_checkpoint),
                "history": {name: [float(value) for value in values] for name, values in history.history.items()},
            }
            save_state()
        finally:
            del model
            tf.keras.backend.clear_session()
            gc.collect()

    invocation_started = time.perf_counter()
    interrupted = False
    try:
        for phase_name, target_epochs, candidate_count, promote_count, patience in PHASES:
            if phase_name in state["completed_phases"]:
                continue
            phase_index = [phase[0] for phase in PHASES].index(phase_name)
            if phase_index == 0:
                active, initial_epoch = state["candidates"], 0
            else:
                active = [candidate for candidate in state["candidates"] if candidate.get("promoted_to") == phase_name]
                initial_epoch = PHASES[phase_index - 1][1]
                if len(active) != candidate_count:
                    raise RuntimeError(f"Promozioni mancanti per {phase_name}: attese {candidate_count}, trovate {len(active)}.")
            print(f"2) {phase_name}: {len(active)} candidate fino a {target_epochs} epoche" +
                  (f", patience={patience}." if patience is not None else "."))
            for position, candidate in enumerate(active, start=1):
                if phase_name in candidate["phases"]:
                    continue
                hp = candidate["hyperparameters"]
                print(f"   {position}/{len(active)} {candidate['candidate_id']}: "
                      f"state={hp['state_hidden_units']}, h={hp['current_hidden_projection_dimension']}/"
                      f"{hp['previous_hidden_projection_dimension']}, fusion={hp['fusion_hidden_units']}")
                train_phase(candidate, phase_name, target_epochs, initial_epoch, patience)
            ranked = sorted(active, key=lambda item: best_score(item, phase_name), reverse=True)
            if promote_count is not None:
                phase_index = [phase[0] for phase in PHASES].index(phase_name)
                next_phase = PHASES[phase_index + 1][0]
                for candidate in ranked[:promote_count]:
                    candidate["promoted_to"] = next_phase
                for candidate in ranked[promote_count:]:
                    candidate["promoted_to"] = None
                    remove_checkpoint(candidate["candidate_id"], phase_name,
                                      f"scartata dopo la promozione {phase_name} -> {next_phase}")
                if phase_index > 0:
                    previous_phase = PHASES[phase_index - 1][0]
                    for candidate in ranked[:promote_count]:
                        remove_checkpoint(candidate["candidate_id"], previous_phase,
                                          f"checkpoint {previous_phase} non piu' necessario")
                    for candidate in ranked[promote_count:]:
                        remove_checkpoint(candidate["candidate_id"], previous_phase,
                                          f"scartata dopo la promozione {phase_name} -> {next_phase}")
                print(f"   Promosse {promote_count}/{len(active)} candidate verso {next_phase}.")
            state["completed_phases"].append(phase_name)
            save_state()
    except KeyboardInterrupt:
        interrupted = True
        save_state()
        write_json_atomically(reports_dir / "interruption_report.json", {
            "message": "Tuning interrotto; riesegui lo stesso comando con --resume.",
            "completed_phases": state["completed_phases"],
        })
        print("\nTuning interrotto. Riprendi con lo stesso comando e --resume.")

    final_phase = PHASES[-1][0]
    finalists = [candidate for candidate in state["candidates"] if final_phase in candidate["phases"]]
    finalists.sort(key=lambda item: best_score(item, final_phase), reverse=True)
    write_json_atomically(reports_dir / "tuning_report.json", {
        "experiment": configuration["experiment"], "interrupted": interrupted,
        "elapsed_seconds_this_invocation": time.perf_counter() - invocation_started,
        "completed_phases": state["completed_phases"], f"finalists_after_{final_phase}": finalists,
        "next_step": {
            "automatic": False,
            "message": "Scegliere manualmente una o piu' finaliste e riaddestrarle da zero fino a 1000 epoche con patience 10.",
            "fixed_hyperparameters": configuration["fixed_hyperparameters"],
        },
    })
    if not interrupted and len(finalists) == PHASES[-1][2]:
        print(f"\nTuning {SEARCH_LABEL}1-{SEARCH_LABEL}3 completato: {len(finalists)} finaliste salvate. Nessuna fase finale automatica eseguita.")
        print(f"Report: {reports_dir / 'tuning_report.json'}")


if __name__ == "__main__":
    main()
