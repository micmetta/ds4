#!/usr/bin/env python3
"""Screen Build 6 feature groups with the same reproducible KNN ablation as Build 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from run_build5_knn_feature_importance import (
    RANDOM_STATE,
    choose_rows,
    evaluate_knn,
    load_array,
    project_dense,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent.parent
BUILD6_ROOT = PIPELINE_ROOT / "build_dataset_6" / "ml"
CHECKPOINT_FILE_NAME = "build6_knn_feature_importance_checkpoint.json"
FINAL_REPORT_FILE_NAME = "build6_knn_feature_importance.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ordina i gruppi Build 6 con ablazione KNN.")
    parser.add_argument("--state-data-dir", type=Path, default=BUILD6_ROOT / "ablations" / "01_current_previous_state" / "arrays")
    parser.add_argument("--current-hidden-data-dir", type=Path, default=BUILD6_ROOT / "ablations" / "05_current_hidden_int8" / "arrays")
    parser.add_argument("--previous-hidden-data-dir", type=Path, default=BUILD6_ROOT / "ablations" / "08_previous_hidden_int8" / "arrays")
    parser.add_argument("--current-kv-data-dir", type=Path, default=BUILD6_ROOT / "ablations" / "09_current_kv_f16" / "arrays")
    parser.add_argument("--previous-kv-data-dir", type=Path, default=BUILD6_ROOT / "ablations" / "10_previous_kv_f16" / "arrays")
    parser.add_argument("--token-position-data-dir", type=Path, default=BUILD6_ROOT / "token_position_inputs" / "arrays")
    parser.add_argument(
        "--expert-transition-probabilities-data-dir",
        type=Path,
        default=BUILD6_ROOT / "transition_features" / "expert_transition_probabilities_v1" / "arrays",
        help="Array 256D precomputati q_(t,l) della feature di transizione degli expert.",
    )
    parser.add_argument("--feature-names", type=Path, default=BUILD6_ROOT / "ablations" / "01_current_previous_state" / "artifacts" / "feature_names.json")
    parser.add_argument("--output-dir", type=Path, default=BUILD6_ROOT / "ablations" / "build6_knn_feature_importance")
    parser.add_argument("--knn-max-train-samples", type=int, default=5_000)
    parser.add_argument("--validation-max-samples", type=int, default=5_000)
    parser.add_argument("--projection-dimensions", type=int, default=32)
    parser.add_argument("--n-neighbors", type=int, default=5)
    return parser.parse_args()


def state_groups(train: sparse.csr_matrix, validation: sparse.csr_matrix, names: list[str]) -> dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix]]:
    prefixes = {
        "current_routing_history_multihot": "routing_history_multihot_",
        "current_layer": "layer_",
        "current_router_logit_expert_id": "router_logit_expert_id_",
        "current_router_logit_values": "router_logit_value_",
        "current_state_kv_statistics": "kv_current_",
        "previous_routing_history_multihot": "previous_routing_history_multihot_",
        "previous_router_logit_expert_id": "previous_router_logit_expert_id_",
        "previous_router_logit_values": "previous_router_logit_value_",
        "previous_state_kv_statistics": "previous_kv_current_",
    }
    result = {}
    for group, prefix in prefixes.items():
        columns = [index for index, name in enumerate(names) if name.startswith(prefix)]
        if not columns:
            raise ValueError(f"Nessuna feature state per {group}.")
        result[group] = train[:, columns].tocsr(), validation[:, columns].tocsr()
    return result


def checkpoint_configuration(args: argparse.Namespace) -> dict[str, int]:
    """Return only the options that change a reproducible KNN result."""
    return {
        "knn_max_train_samples": args.knn_max_train_samples,
        "validation_max_samples": args.validation_max_samples,
        "projection_dimensions": args.projection_dimensions,
        "n_neighbors": args.n_neighbors,
        "random_state": RANDOM_STATE,
    }


def save_checkpoint(
    path: Path,
    configuration: dict[str, int],
    full_representation: dict | None,
    results: list[dict],
) -> None:
    """Persist enough information to continue after an interrupted ablation."""
    write_json(path, {
        "experiment": "build6_knn_leave_one_group_out_importance",
        "status": "in_progress",
        "configuration": configuration,
        "full_representation": full_representation,
        "completed_ablations": results,
    })


def main() -> None:
    args = parse_args()
    reports_dir = args.output_dir / "reports"
    report_path = reports_dir / FINAL_REPORT_FILE_NAME
    checkpoint_path = reports_dir / CHECKPOINT_FILE_NAME
    if report_path.exists():
        raise FileExistsError(f"Report finale gia' esistente: {report_path}")
    configuration = checkpoint_configuration(args)
    checkpoint: dict | None = None
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("configuration") != configuration:
            raise ValueError(
                "Checkpoint KNN esistente creato con opzioni diverse. "
                "Usare lo stesso comando oppure scegliere un'altra --output-dir."
            )
        print(f"   Riprendo dal checkpoint: {len(checkpoint.get('completed_ablations', []))} ablazioni gia' completate.")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output non vuoto senza checkpoint: {args.output_dir}. "
            "Scegliere una nuova directory oppure rimuovere solo l'output incompleto."
        )
    print("1) Carico i sottoinsiemi riproducibili Build 6 per KNN...")
    full_train = sparse.load_npz(args.state_data_dir / "X_classic_train.npz").tocsr()
    full_validation = sparse.load_npz(args.state_data_dir / "X_classic_validation.npz").tocsr()
    train_rows = choose_rows(full_train.shape[0], args.knn_max_train_samples, 0)
    validation_rows = choose_rows(full_validation.shape[0], args.validation_max_samples, 1)
    y_train = np.asarray(load_array(args.state_data_dir, "y", "train")[train_rows], dtype=np.uint8)
    y_validation = np.asarray(load_array(args.state_data_dir, "y", "validation")[validation_rows], dtype=np.uint8)
    names = json.loads(args.feature_names.read_text(encoding="utf-8"))["feature_names"]
    if len(names) != full_train.shape[1]:
        raise ValueError("feature_names.json non allineato allo state Build 6.")
    groups = state_groups(full_train[train_rows], full_validation[validation_rows], names)
    dense_sources = (
        ("current_raw_hidden_4096d", args.current_hidden_data_dir, "raw_hidden_normalized_int8", 100),
        ("previous_raw_hidden_4096d", args.previous_hidden_data_dir, "raw_hidden_normalized_int8", 200),
        ("current_kv_512d", args.current_kv_data_dir, "kv_normalized", 300),
        ("previous_kv_512d", args.previous_kv_data_dir, "kv_normalized", 400),
        (
            "expert_transition_probabilities_256d",
            args.expert_transition_probabilities_data_dir,
            "expert_transition_probabilities",
            500,
        ),
    )
    for name, directory, array_name, seed in dense_sources:
        groups[name] = project_dense(load_array(directory, array_name, "train")[train_rows], load_array(directory, array_name, "validation")[validation_rows], args.projection_dimensions, seed)
    groups["token_position"] = (
        sparse.csr_matrix(load_array(args.token_position_data_dir, "token_position", "train")[train_rows]),
        sparse.csr_matrix(load_array(args.token_position_data_dir, "token_position", "validation")[validation_rows]),
    )
    print(f"2) Valuto KNN completo e ablazione di {len(groups)} gruppi...")
    group_names = list(groups)
    full_representation = checkpoint.get("full_representation") if checkpoint else None
    results = list(checkpoint.get("completed_ablations", [])) if checkpoint else []
    completed_groups = {item["removed_group"] for item in results}
    if full_representation is None:
        full_recall, full_dimensions, full_seconds = evaluate_knn(
            [groups[name][0] for name in group_names],
            [groups[name][1] for name in group_names],
            y_train,
            y_validation,
            args.n_neighbors,
        )
        full_representation = {
            "validation_recall_at_6": full_recall,
            "feature_dimensions": full_dimensions,
            "elapsed_seconds": full_seconds,
        }
        save_checkpoint(checkpoint_path, configuration, full_representation, results)
        print(f"   completo: Recall@6 validation = {full_recall:.6f}")
    else:
        full_recall = float(full_representation["validation_recall_at_6"])
        print(f"   completo dal checkpoint: Recall@6 validation = {full_recall:.6f}")
    for name in group_names:
        if name in completed_groups:
            continue
        included = [other for other in group_names if other != name]
        recall, dimensions, elapsed = evaluate_knn([groups[other][0] for other in included], [groups[other][1] for other in included], y_train, y_validation, args.n_neighbors)
        results.append({"removed_group": name, "validation_recall_at_6": recall, "importance_recall_drop": full_recall - recall, "feature_dimensions_remaining": dimensions, "elapsed_seconds": elapsed})
        save_checkpoint(checkpoint_path, configuration, full_representation, results)
        print(f"   senza {name}: {recall:.6f} (delta {full_recall - recall:+.6f})")
    results.sort(key=lambda item: item["importance_recall_drop"], reverse=True)
    write_json(report_path, {
        "experiment": "build6_knn_leave_one_group_out_importance", "interpretation": "Screening KNN del segnale predittivo; non importanza dei pesi FFN.",
        "knn": {"n_neighbors": args.n_neighbors, "weights": "distance", "train_examples": len(train_rows), "validation_examples": len(validation_rows), "projection_dimensions_for_dense_groups": args.projection_dimensions, "random_state": RANDOM_STATE},
        "full_representation": full_representation, "results_ranked_by_importance": results,
    })
    checkpoint_path.unlink(missing_ok=True)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
