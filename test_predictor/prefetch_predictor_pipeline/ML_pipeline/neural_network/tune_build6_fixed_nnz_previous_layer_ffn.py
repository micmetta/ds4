#!/usr/bin/env python3
"""Controlled three-stage successive-halving tuning for the Build 6 FFN."""

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
    build_fixed_nnz_state_two_hidden_two_kv_ffn,
    load_expert_transition_probabilities_inputs,
    load_fixed_nnz_state_inputs,
    load_labels,
    load_materialized_kv_inputs,
    load_quantized_hidden_inputs,
    make_fixed_nnz_state_two_hidden_two_kv_dataset,
    require_tensorflow,
)


STATE_HIDDEN_UNITS = (512, 1024, 2048)
HIDDEN_PROJECTION_DIMENSIONS = (64, 512, 1024)
KV_PROJECTION_CHOICES = ("direct", "64", "128", "256")
LEARNING_RATES = (0.00003, 0.00005, 0.0001, 0.00015)
FUSION_HIDDEN_UNITS = (0, 512, 1024, 2048)
# Historical tuner: None keeps the original architecture unchanged.  Dedicated
# wrappers can enable a transition-probability branch without duplicating the
# reliable resume and successive-halving logic below.
EXPERT_TRANSITION_PROJECTION_CHOICES: tuple[str, ...] | None = None
PHASES = (
    ("phase_03", 3, 30, 15),
    ("phase_07", 7, 15, 5),
    ("phase_20", 20, 5, None),
)


def write_json_atomically(path: Path, content: Any) -> None:
    """Write one complete JSON state file without a partially written target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(content, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_positions(directory: Path, split: str, examples: int) -> np.ndarray:
    path = directory / "arrays" / f"token_position_{split}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"token_position non trovata: {path}")
    values = np.load(path, mmap_mode="r")
    if values.shape != (examples, 1) or values.dtype != np.float32:
        raise ValueError(f"token_position incompatibile: {path}")
    return values


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selezione 30 -> 15 -> 5 della FFN Build 6 con rami separati."
    )
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--current-kv-data-dir", type=Path, required=True)
    parser.add_argument("--previous-kv-data-dir", type=Path, required=True)
    parser.add_argument("--token-position-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path,
                        help="Root della feature 256D materializzata; omesso = architettura storica.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Riprende dalla prima fase non completata.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0,
                        help="Lambda della loss ranking; 0 mantiene la BCE storica.")
    parser.add_argument("--ranking-loss-margin", type=float, default=0.05,
                        help="Margine minimo target/non-target della loss ranking.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--shuffle-block-size", type=int, default=65536)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    # These labels are intentionally cosmetic: wrappers for later dataset
    # builds reuse the proven tuning implementation without changing the
    # Build 6 defaults or mixing reports from distinct experiments.
    parser.add_argument("--experiment-name", default="build6_successive_halving_previous_layer_ffn")
    parser.add_argument("--dataset-label", default="Build 6")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if (
        min(args.batch_size, args.shuffle_block_size) < 1
        or args.shuffle_block_size < args.batch_size
        or args.early_stopping_patience < 0
        or not 0 <= args.dropout < 1
        or args.l2_regularization < 0
        or args.ranking_loss_weight < 0
        or args.ranking_loss_margin <= 0
    ):
        raise ValueError("Parametri del tuning non validi.")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"La cartella output esiste gia': {args.output_dir}. Usa --resume oppure una nuova directory."
        )
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError(f"Non esiste alcun tuning da riprendere: {args.output_dir}")
    if EXPERT_TRANSITION_PROJECTION_CHOICES is None and args.expert_transition_probabilities_data_dir is not None:
        raise ValueError("Questo tuner storico non definisce lo spazio per expert_transition_probabilities.")
    if EXPERT_TRANSITION_PROJECTION_CHOICES is not None and args.expert_transition_probabilities_data_dir is None:
        raise ValueError("La variante con expert_transition_probabilities richiede il relativo data directory.")


def search_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Describe the immutable experiment, including the exact stage schedule."""
    configuration = {
        "data": {
            "state_data_dir": str(args.state_data_dir),
            "fixed_nnz_state_data_dir": str(args.fixed_nnz_state_data_dir),
            "current_quantized_hidden_data_dir": str(args.current_quantized_hidden_data_dir),
            "previous_quantized_hidden_data_dir": str(args.previous_quantized_hidden_data_dir),
            "current_kv_data_dir": str(args.current_kv_data_dir),
            "previous_kv_data_dir": str(args.previous_kv_data_dir),
            "token_position_data_dir": str(args.token_position_data_dir),
        },
        "search_space": {
            "state_hidden_units": list(STATE_HIDDEN_UNITS),
            "current_raw_hidden_projection_dimension": list(HIDDEN_PROJECTION_DIMENSIONS),
            "previous_raw_hidden_projection_dimension": list(HIDDEN_PROJECTION_DIMENSIONS),
            "current_kv_projection": list(KV_PROJECTION_CHOICES),
            "previous_kv_projection": list(KV_PROJECTION_CHOICES),
            "fusion_hidden_units": list(FUSION_HIDDEN_UNITS),
            "learning_rate": list(LEARNING_RATES),
        },
        "selection_schedule": [
            {"name": name, "epochs": epochs, "candidates": candidates, "promote": promote}
            for name, epochs, candidates, promote in PHASES
        ],
        "fixed_hyperparameters": {
            "activation": "relu",
            "dropout": args.dropout,
            "l2_regularization": args.l2_regularization,
            "ranking_loss_weight": args.ranking_loss_weight,
            "ranking_loss_margin": args.ranking_loss_margin,
            "batch_size": args.batch_size,
            "shuffle_block_size": args.shuffle_block_size,
            "early_stopping_patience": args.early_stopping_patience,
            "policy_tensorflow": "float32",
            "token_embedding_input": False,
        },
        "objective": "massimizzare il migliore val_recall_at_6 osservato in ogni fase",
        "random_state": RANDOM_STATE,
    }
    if args.expert_transition_probabilities_data_dir is not None:
        configuration["data"]["expert_transition_probabilities_data_dir"] = str(
            args.expert_transition_probabilities_data_dir
        )
        configuration["search_space"]["expert_transition_projection"] = list(
            EXPERT_TRANSITION_PROJECTION_CHOICES or ()
        )
    return configuration


def sample_candidates() -> list[dict[str, Any]]:
    """Choose 30 distinct architectures reproducibly from the complete space."""
    dimensions = [
        STATE_HIDDEN_UNITS,
        HIDDEN_PROJECTION_DIMENSIONS,
        HIDDEN_PROJECTION_DIMENSIONS,
        KV_PROJECTION_CHOICES,
        KV_PROJECTION_CHOICES,
        FUSION_HIDDEN_UNITS,
        LEARNING_RATES,
    ]
    if EXPERT_TRANSITION_PROJECTION_CHOICES is not None:
        dimensions.append(EXPERT_TRANSITION_PROJECTION_CHOICES)
    complete_space = list(itertools.product(*dimensions))
    selected = np.random.default_rng(RANDOM_STATE).choice(len(complete_space), size=30, replace=False)
    candidates = []
    for number, index in enumerate(selected, start=1):
        values = complete_space[int(index)]
        state_units, current_hidden, previous_hidden, current_kv, previous_kv, fusion, learning_rate = values[:7]
        candidates.append({
            "candidate_id": f"candidate_{number:02d}",
            "hyperparameters": {
                "state_hidden_units": state_units,
                "current_raw_hidden_projection_dimension": current_hidden,
                "previous_raw_hidden_projection_dimension": previous_hidden,
                "current_kv_projection": current_kv,
                "previous_kv_projection": previous_kv,
                "fusion_hidden_units": fusion,
                "learning_rate": learning_rate,
            },
        })
        if EXPERT_TRANSITION_PROJECTION_CHOICES is not None:
            candidates[-1]["hyperparameters"]["expert_transition_projection"] = values[7]
    return candidates


def phase_checkpoint_path(root: Path, candidate_id: str, phase_name: str) -> Path:
    return root / "candidate_checkpoints" / candidate_id / f"{phase_name}.model_and_adam.npz"


def save_model_and_adam(path: Path, model) -> None:
    """Save one exact promotion point: model tensors and all Adam variables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    payload = {
        **{f"model_{index}": value for index, value in enumerate(model.get_weights())},
        **{
            f"adam_{index}": variable.numpy()
            for index, variable in enumerate(model.optimizer.variables)
        },
    }
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def load_model_and_adam(path: Path, model) -> None:
    """Restore the model and Adam exactly as they were at a promotion point."""
    with np.load(path) as archive:
        model_keys = sorted(
            (key for key in archive.files if key.startswith("model_")),
            key=lambda key: int(key.removeprefix("model_")),
        )
        adam_keys = sorted(
            (key for key in archive.files if key.startswith("adam_")),
            key=lambda key: int(key.removeprefix("adam_")),
        )
        weights = [archive[key] for key in model_keys]
        adam_values = [archive[key] for key in adam_keys]
    model.set_weights(weights)
    # Keras creates Adam's momentum and velocity slots lazily. Build them
    # before assigning the saved values so the variable order is stable.
    model.optimizer.build(model.trainable_variables)
    adam_variables = list(model.optimizer.variables)
    if len(adam_values) != len(adam_variables):
        raise ValueError(
            f"Checkpoint Adam incompatibile: {len(adam_values)} variabili salvate, "
            f"{len(adam_variables)} create dal modello."
        )
    for variable, value in zip(adam_variables, adam_values, strict=True):
        if tuple(variable.shape) != tuple(value.shape):
            raise ValueError("Checkpoint Adam incompatibile: shape di una variabile diversa.")
        variable.assign(value)


def best_phase_score(candidate: dict[str, Any], phase_name: str) -> float:
    return float(candidate["phases"][phase_name]["best_validation_recall_at_6"])


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = search_configuration(args)
    reports_directory = args.output_dir / "reports"
    configuration_path = reports_directory / "search_configuration.json"
    state_path = reports_directory / "tuning_state.json"
    if args.resume:
        if not configuration_path.is_file() or not state_path.is_file():
            raise FileNotFoundError("Mancano configurazione o stato: non e' sicuro riprendere il tuning.")
        if json.loads(configuration_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("La configurazione richiesta differisce da quella salvata.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        write_json_atomically(configuration_path, configuration)
        state = {
            "experiment": args.experiment_name,
            "candidates": sample_candidates(),
            "completed_phases": [],
            "checkpoint_cleanup": [],
        }
        for candidate in state["candidates"]:
            candidate["phases"] = {}
        write_json_atomically(state_path, state)

    tf = require_tensorflow()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    print(f"1) Carico gli array {args.dataset_label} per train e validation...")
    y_train = load_labels(args.state_data_dir, "train")
    y_validation = load_labels(args.state_data_dir, "validation")
    train_indices, train_values, metadata = load_fixed_nnz_state_inputs(args.fixed_nnz_state_data_dir, "train", y_train.shape[0])
    validation_indices, validation_values, validation_metadata = load_fixed_nnz_state_inputs(args.fixed_nnz_state_data_dir, "validation", y_validation.shape[0])
    if metadata != validation_metadata:
        raise ValueError("Train e validation usano metadata state fixed-NNZ diverse.")
    train_position = load_positions(args.token_position_data_dir, "train", y_train.shape[0])
    validation_position = load_positions(args.token_position_data_dir, "validation", y_validation.shape[0])
    train_current_hidden, current_scale = load_quantized_hidden_inputs(args.current_quantized_hidden_data_dir, "train", y_train.shape[0])
    validation_current_hidden, validation_current_scale = load_quantized_hidden_inputs(args.current_quantized_hidden_data_dir, "validation", y_validation.shape[0])
    train_previous_hidden, previous_scale = load_quantized_hidden_inputs(args.previous_quantized_hidden_data_dir, "train", y_train.shape[0])
    validation_previous_hidden, validation_previous_scale = load_quantized_hidden_inputs(args.previous_quantized_hidden_data_dir, "validation", y_validation.shape[0])
    if current_scale != validation_current_scale or previous_scale != validation_previous_scale:
        raise ValueError("Train e validation usano scale hidden int8 diverse.")
    train_current_kv = load_materialized_kv_inputs(args.current_kv_data_dir, "train", y_train.shape[0])
    validation_current_kv = load_materialized_kv_inputs(args.current_kv_data_dir, "validation", y_validation.shape[0])
    train_previous_kv = load_materialized_kv_inputs(args.previous_kv_data_dir, "train", y_train.shape[0])
    validation_previous_kv = load_materialized_kv_inputs(args.previous_kv_data_dir, "validation", y_validation.shape[0])
    train_transition = validation_transition = None
    if args.expert_transition_probabilities_data_dir is not None:
        train_transition = load_expert_transition_probabilities_inputs(
            args.expert_transition_probabilities_data_dir, "train", y_train.shape[0]
        )
        validation_transition = load_expert_transition_probabilities_inputs(
            args.expert_transition_probabilities_data_dir, "validation", y_validation.shape[0]
        )
    train_steps = math.ceil(y_train.shape[0] / args.batch_size)
    validation_steps = math.ceil(y_validation.shape[0] / args.batch_size)
    train_dataset = make_fixed_nnz_state_two_hidden_two_kv_dataset(
        train_indices, train_values, y_train, train_position, train_current_hidden,
        train_previous_hidden, train_current_kv, train_previous_kv, args.batch_size,
        True, tf, args.shuffle_block_size, train_transition,
    ).repeat()
    validation_dataset = make_fixed_nnz_state_two_hidden_two_kv_dataset(
        validation_indices, validation_values, y_validation, validation_position,
        validation_current_hidden, validation_previous_hidden, validation_current_kv,
        validation_previous_kv, args.batch_size, False, tf, args.shuffle_block_size, validation_transition,
    ).repeat()

    progress_directory = args.output_dir / "candidate_progress"

    class BestWeightsCheckpoint(tf.keras.callbacks.Callback):
        """Keep the best global phase score, including the score of the previous phase."""

        def __init__(self, path: Path, initial_best: float) -> None:
            super().__init__()
            self.path = path
            self.best = initial_best

        def on_epoch_end(self, epoch, logs=None) -> None:
            value = float((logs or {}).get("val_recall_at_6", float("-inf")))
            if value > self.best:
                self.best = value
                save_model_and_adam(self.path, self.model)

    class PhaseProgress(tf.keras.callbacks.Callback):
        """Write every completed epoch of one candidate and one selection phase."""

        def __init__(self, candidate_id: str, phase_name: str, hyperparameters: dict[str, Any]) -> None:
            super().__init__()
            self.path = progress_directory / candidate_id / f"{phase_name}.jsonl"
            self.candidate_id = candidate_id
            self.phase_name = phase_name
            self.hyperparameters = hyperparameters
            self.started_at: float | None = None

        def on_epoch_begin(self, epoch, logs=None) -> None:
            self.started_at = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None) -> None:
            record = {
                "candidate_id": self.candidate_id,
                "phase": self.phase_name,
                "epoch": int(epoch) + 1,
                "hyperparameters": self.hyperparameters,
                "epoch_seconds": None if self.started_at is None else time.perf_counter() - self.started_at,
                "metrics": {name: float(value) for name, value in (logs or {}).items()},
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()

    def build_model(hyperparameters: dict[str, Any]):
        current_kv = hyperparameters["current_kv_projection"]
        previous_kv = hyperparameters["previous_kv_projection"]
        fusion_units = hyperparameters["fusion_hidden_units"]
        return build_fixed_nnz_state_two_hidden_two_kv_ffn(
            metadata["feature_count"], metadata["input_slots"], hyperparameters["state_hidden_units"],
            [] if fusion_units == 0 else [fusion_units], "relu", args.dropout,
            hyperparameters["learning_rate"], args.l2_regularization, tf,
            hyperparameters["current_raw_hidden_projection_dimension"],
            hyperparameters["previous_raw_hidden_projection_dimension"], current_scale,
            previous_scale, None if current_kv == "direct" else int(current_kv),
            None if previous_kv == "direct" else int(previous_kv),
            args.ranking_loss_weight, args.ranking_loss_margin,
            None if "expert_transition_projection" not in hyperparameters
            else (0 if hyperparameters["expert_transition_projection"] == "direct"
                  else int(hyperparameters["expert_transition_projection"])),
        )

    def save_state() -> None:
        write_json_atomically(state_path, state)

    def remove_checkpoint(candidate_id: str, phase_name: str, reason: str) -> None:
        """Delete only a checkpoint that is no longer needed for any resume path."""
        path = phase_checkpoint_path(args.output_dir, candidate_id, phase_name)
        if not path.is_file():
            return
        path.unlink()
        state.setdefault("checkpoint_cleanup", []).append({
            "candidate_id": candidate_id,
            "phase": phase_name,
            "reason": reason,
        })
        # Candidate directories have no useful file besides phase checkpoints.
        try:
            path.parent.rmdir()
        except OSError:
            pass

    def train_phase(candidate: dict[str, Any], phase_name: str, target_epochs: int, start_epoch: int) -> None:
        candidate_id = candidate["candidate_id"]
        hyperparameters = candidate["hyperparameters"]
        phase_checkpoint = phase_checkpoint_path(args.output_dir, candidate_id, phase_name)
        # A retry of an interrupted active phase starts from the stable checkpoint
        # of the preceding phase, so it cannot corrupt a completed promotion.
        phase_checkpoint.unlink(missing_ok=True)
        model = None
        try:
            model = build_model(hyperparameters)
            previous_best = float("-inf")
            if start_epoch:
                previous_phase = "phase_03" if start_epoch == 3 else "phase_07"
                previous_record = candidate["phases"][previous_phase]
                previous_checkpoint = Path(previous_record["checkpoint_path"])
                load_model_and_adam(previous_checkpoint, model)
                previous_best = float(previous_record["best_validation_recall_at_6"])
                save_model_and_adam(phase_checkpoint, model)
            checkpoint = BestWeightsCheckpoint(phase_checkpoint, previous_best)
            callbacks = [
                PhaseProgress(candidate_id, phase_name, hyperparameters),
                checkpoint,
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_recall_at_6", mode="max", patience=args.early_stopping_patience,
                    restore_best_weights=False,
                ),
            ]
            started_at = time.perf_counter()
            history = model.fit(
                train_dataset, steps_per_epoch=train_steps, validation_data=validation_dataset,
                validation_steps=validation_steps, initial_epoch=start_epoch, epochs=target_epochs,
                callbacks=callbacks, shuffle=False, verbose=2,
            )
            if not phase_checkpoint.is_file():
                # The first validation epoch is always a valid best if there was no previous phase.
                save_model_and_adam(phase_checkpoint, model)
            phase_scores = [float(value) for value in history.history.get("val_recall_at_6", [])]
            observed_best = max(phase_scores, default=float("-inf"))
            candidate["phases"][phase_name] = {
                "status": "completed",
                "target_epochs": target_epochs,
                "initial_epoch": start_epoch,
                "epochs_executed_in_phase": len(phase_scores),
                "best_validation_recall_at_6": max(previous_best, observed_best),
                "best_validation_recall_at_6_in_phase": observed_best,
                "elapsed_seconds": time.perf_counter() - started_at,
                "checkpoint_path": str(phase_checkpoint),
                "history": {name: [float(value) for value in values] for name, values in history.history.items()},
            }
            save_state()
        finally:
            # Ogni candidata costruisce un grafo e cache TensorFlow propri. Senza
            # questo rilascio resterebbero residenti e rallenterebbero le candidate
            # successive, pur non essendo piu' necessari dopo il checkpoint.
            del model
            tf.keras.backend.clear_session()
            gc.collect()

    started_at = time.perf_counter()
    interrupted = False
    try:
        for phase_name, target_epochs, candidate_count, promote_count in PHASES:
            if phase_name in state["completed_phases"]:
                continue
            if phase_name == "phase_03":
                active = state["candidates"]
                start_epoch = 0
            else:
                previous_phase = "phase_03" if phase_name == "phase_07" else "phase_07"
                active = [candidate for candidate in state["candidates"] if candidate.get("promoted_to") == phase_name]
                start_epoch = 3 if phase_name == "phase_07" else 7
                if len(active) != candidate_count:
                    raise RuntimeError(f"Promozioni mancanti per {phase_name}: attese {candidate_count}, trovate {len(active)}.")
            print(f"2) {phase_name}: porto {len(active)} configurazioni fino a {target_epochs} epoche...")
            for number, candidate in enumerate(active, start=1):
                if phase_name in candidate["phases"]:
                    continue
                print(f"   {number}/{len(active)} {candidate['candidate_id']}")
                train_phase(candidate, phase_name, target_epochs, start_epoch)
            ranked = sorted(active, key=lambda item: best_phase_score(item, phase_name), reverse=True)
            if promote_count is not None:
                next_phase = "phase_07" if phase_name == "phase_03" else "phase_20"
                for candidate in ranked[:promote_count]:
                    candidate["promoted_to"] = next_phase
                for candidate in ranked[promote_count:]:
                    candidate["promoted_to"] = None
                if phase_name == "phase_03":
                    for candidate in ranked[promote_count:]:
                        remove_checkpoint(
                            candidate["candidate_id"], "phase_03",
                            "scartata dopo la promozione 30 -> 15",
                        )
                else:
                    for candidate in ranked[promote_count:]:
                        remove_checkpoint(
                            candidate["candidate_id"], "phase_03",
                            "scartata dopo la promozione 15 -> 5",
                        )
                        remove_checkpoint(
                            candidate["candidate_id"], "phase_07",
                            "scartata dopo la promozione 15 -> 5",
                        )
                    for candidate in ranked[:promote_count]:
                        remove_checkpoint(
                            candidate["candidate_id"], "phase_03",
                            "phase_07 completata: il checkpoint phase_03 non serve piu'",
                        )
                print(f"   Promosse {promote_count}/{len(active)} configurazioni verso {next_phase}.")
            else:
                for candidate in ranked:
                    remove_checkpoint(
                        candidate["candidate_id"], "phase_03",
                        "phase_20 completata: il checkpoint phase_03 non serve piu'",
                    )
                    remove_checkpoint(
                        candidate["candidate_id"], "phase_07",
                        "phase_20 completata: il checkpoint phase_07 non serve piu'",
                    )
            state["completed_phases"].append(phase_name)
            save_state()
    except KeyboardInterrupt:
        interrupted = True
        save_state()
        write_json_atomically(reports_directory / "interruption_report.json", {
            "interrupted_at_unix_seconds": time.time(),
            "message": "Tuning interrotto dall'utente; le fasi e le candidate gia' completate restano salvate.",
            "resume_command_requires": "lo stesso comando con --resume",
        })
        print("\nTuning interrotto: stato scritto. Riprendi con lo stesso comando e --resume.")

    completed_finalists = [
        candidate for candidate in state["candidates"] if "phase_20" in candidate["phases"]
    ]
    completed_finalists.sort(key=lambda item: best_phase_score(item, "phase_20"), reverse=True)
    report_path = reports_directory / "tuning_report.json"
    write_json_atomically(report_path, {
        "experiment": args.experiment_name,
        "interrupted": interrupted,
        "elapsed_seconds_this_invocation": time.perf_counter() - started_at,
        "completed_phases": state["completed_phases"],
        "finalists_after_20_epochs": completed_finalists,
        "phase_2": {"epochs": 1000, "early_stopping_patience": 5},
    })
    if not interrupted and len(completed_finalists) == 5:
        print(f"\nSelezione {args.dataset_label} completata: 5 finaliste portate fino a 20 epoche.")
        print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
