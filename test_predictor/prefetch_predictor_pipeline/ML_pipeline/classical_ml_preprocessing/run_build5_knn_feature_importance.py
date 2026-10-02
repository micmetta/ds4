#!/usr/bin/env python3
"""Rank Build 5 feature groups with a fast, reproducible KNN ablation.

This is a *predictive-signal* screening, not an explanation of the neural
network weights.  Every KNN receives the same train/validation row subsets.
The importance of one group is the validation Recall@6 loss after removing it
from the full Build 5 representation.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier

from train_classical_models import TOP_K, calcola_recall_at_6, estrai_score_positivi


RANDOM_STATE = 42
NUM_EXPERTS = 256
PROJECTION_DIMENSIONS = 32

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent.parent
BUILD5_ROOT = PIPELINE_ROOT / "build_dataset_5" / "ml"
DEFAULT_STATE_DIR = BUILD5_ROOT / "ablations" / "09_routing_layer_router_state_kv_current" / "arrays"
DEFAULT_HIDDEN_DIR = BUILD5_ROOT / "ablations" / "13_routing_layer_router_state_kv_current_materialized_raw_hidden_int8" / "arrays"
DEFAULT_KV_DIR = BUILD5_ROOT / "ablations" / "14_routing_layer_router_state_kv_current_materialized_kv_f16" / "arrays"
DEFAULT_TOKEN_DIR = BUILD5_ROOT / "ablations" / "10_routing_layer_router_state_kv_current_token_embedding" / "arrays"
DEFAULT_TOKEN_EMBEDDING_DIR = BUILD5_ROOT / "token_embedding_artifacts"
DEFAULT_FEATURE_NAMES = BUILD5_ROOT / "ablations" / "09_routing_layer_router_state_kv_current" / "artifacts" / "feature_names.json"
DEFAULT_OUTPUT_DIR = BUILD5_ROOT / "ablations" / "build5_knn_feature_importance"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ordina i gruppi di feature Build 5 con ablazione KNN."
    )
    parser.add_argument("--state-data-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--quantized-hidden-data-dir", type=Path, default=DEFAULT_HIDDEN_DIR)
    parser.add_argument("--kv-data-dir", type=Path, default=DEFAULT_KV_DIR)
    parser.add_argument("--token-data-dir", type=Path, default=DEFAULT_TOKEN_DIR)
    parser.add_argument("--token-embedding-dir", type=Path, default=DEFAULT_TOKEN_EMBEDDING_DIR)
    parser.add_argument("--feature-names", type=Path, default=DEFAULT_FEATURE_NAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--knn-max-train-samples", type=int, default=5_000)
    parser.add_argument("--validation-max-samples", type=int, default=5_000)
    parser.add_argument("--projection-dimensions", type=int, default=PROJECTION_DIMENSIONS)
    parser.add_argument("--n-neighbors", type=int, default=5)
    return parser.parse_args()


def load_array(directory: Path, name: str, split: str) -> np.ndarray:
    path = directory / f"{name}_{split}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Array mancante: {path}")
    return np.load(path, mmap_mode="r")


def choose_rows(rows: int, maximum: int, seed_offset: int) -> np.ndarray:
    if maximum < 1:
        raise ValueError("I limiti degli esempi KNN devono essere positivi.")
    if maximum >= rows:
        return np.arange(rows, dtype=np.int64)
    generator = np.random.default_rng(RANDOM_STATE + seed_offset)
    return np.sort(generator.choice(rows, size=maximum, replace=False))


def project_dense(
    train: np.ndarray, validation: np.ndarray, dimensions: int, seed: int
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Keep all dimensions for zero, otherwise project, then standardize."""
    if dimensions < 0:
        raise ValueError("--projection-dimensions deve essere zero oppure positivo.")
    if dimensions == 0:
        train_projected = np.asarray(train, dtype=np.float32)
        validation_projected = np.asarray(validation, dtype=np.float32)
    else:
        generator = np.random.default_rng(seed)
        matrix = generator.normal(
            0.0, 1.0 / np.sqrt(train.shape[1]), size=(train.shape[1], dimensions)
        ).astype(np.float32)
        train_projected = np.asarray(train, dtype=np.float32) @ matrix
        validation_projected = np.asarray(validation, dtype=np.float32) @ matrix
    mean = train_projected.mean(axis=0, keepdims=True)
    std = train_projected.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (
        sparse.csr_matrix((train_projected - mean) / std, dtype=np.float32),
        sparse.csr_matrix((validation_projected - mean) / std, dtype=np.float32),
    )


def state_groups(
    state_train: sparse.csr_matrix,
    state_validation: sparse.csr_matrix,
    feature_names: list[str],
) -> dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix]]:
    prefixes = {
        "routing_history_multihot": "routing_history_multihot_",
        "layer": "layer_",
        "router_logit_expert_id": "router_logit_expert_id_",
        "router_logit_values": "router_logit_value_",
        "state_kv_statistics": "kv_current_",
    }
    groups = {}
    for name, prefix in prefixes.items():
        columns = [index for index, feature in enumerate(feature_names) if feature.startswith(prefix)]
        if not columns:
            raise ValueError(f"Nessuna colonna state trovata per il gruppo {name}.")
        groups[name] = (state_train[:, columns].tocsr(), state_validation[:, columns].tocsr())
    return groups


def evaluate_knn(
    train_parts: list[sparse.csr_matrix],
    validation_parts: list[sparse.csr_matrix],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    n_neighbors: int,
) -> tuple[float, int, float]:
    x_train = sparse.hstack(train_parts, format="csr", dtype=np.float32)
    x_validation = sparse.hstack(validation_parts, format="csr", dtype=np.float32)
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)
    started_at = time.perf_counter()
    model.fit(x_train, y_train)
    scores = estrai_score_positivi(model, x_validation)
    return calcola_recall_at_6(scores, y_validation), int(x_train.shape[1]), time.perf_counter() - started_at


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    print("1) Carico i sottoinsiemi riproducibili Build 5 per KNN...")
    state_train_full = sparse.load_npz(args.state_data_dir / "X_classic_train.npz").tocsr()
    state_validation_full = sparse.load_npz(args.state_data_dir / "X_classic_validation.npz").tocsr()
    train_rows = choose_rows(state_train_full.shape[0], args.knn_max_train_samples, 0)
    validation_rows = choose_rows(state_validation_full.shape[0], args.validation_max_samples, 1)
    y_train = np.asarray(load_array(args.state_data_dir, "y", "train")[train_rows], dtype=np.uint8)
    y_validation = np.asarray(load_array(args.state_data_dir, "y", "validation")[validation_rows], dtype=np.uint8)
    state_train = state_train_full[train_rows]
    state_validation = state_validation_full[validation_rows]

    feature_names = json.loads(args.feature_names.read_text(encoding="utf-8"))["feature_names"]
    if len(feature_names) != state_train.shape[1]:
        raise ValueError("feature_names.json non corrisponde alle colonne di X_classic.")
    groups = state_groups(state_train, state_validation, feature_names)

    raw_train = load_array(args.quantized_hidden_data_dir, "raw_hidden_normalized_int8", "train")[train_rows]
    raw_validation = load_array(args.quantized_hidden_data_dir, "raw_hidden_normalized_int8", "validation")[validation_rows]
    groups["raw_hidden_4096d"] = project_dense(raw_train, raw_validation, args.projection_dimensions, 100)

    token_indices_train = load_array(args.token_data_dir, "token_embedding_index", "train")[train_rows]
    token_indices_validation = load_array(args.token_data_dir, "token_embedding_index", "validation")[validation_rows]
    embeddings = np.load(args.token_embedding_dir / "token_embeddings_f16.npy", mmap_mode="r")
    groups["frozen_token_embedding_4096d"] = project_dense(
        embeddings[token_indices_train], embeddings[token_indices_validation], args.projection_dimensions, 200
    )

    kv_train = load_array(args.kv_data_dir, "kv_normalized", "train")[train_rows]
    kv_validation = load_array(args.kv_data_dir, "kv_normalized", "validation")[validation_rows]
    groups["kv_current_512d"] = project_dense(kv_train, kv_validation, args.projection_dimensions, 300)

    position_train = np.asarray(load_array(args.token_data_dir, "token_position", "train")[train_rows], dtype=np.float32)
    position_validation = np.asarray(load_array(args.token_data_dir, "token_position", "validation")[validation_rows], dtype=np.float32)
    groups["token_position"] = (sparse.csr_matrix(position_train), sparse.csr_matrix(position_validation))

    print(f"2) Valuto KNN completo e ablazione di {len(groups)} gruppi...")
    group_names = list(groups)
    full_recall, full_dimensions, full_seconds = evaluate_knn(
        [groups[name][0] for name in group_names],
        [groups[name][1] for name in group_names],
        y_train, y_validation, args.n_neighbors,
    )
    print(f"   completo: Recall@6 validation = {full_recall:.6f}")
    results = []
    for name in group_names:
        included = [other for other in group_names if other != name]
        recall, dimensions, seconds = evaluate_knn(
            [groups[other][0] for other in included],
            [groups[other][1] for other in included],
            y_train, y_validation, args.n_neighbors,
        )
        result = {
            "removed_group": name,
            "validation_recall_at_6": recall,
            "importance_recall_drop": full_recall - recall,
            "feature_dimensions_remaining": dimensions,
            "elapsed_seconds": seconds,
        }
        results.append(result)
        print(f"   senza {name}: {recall:.6f} (delta {full_recall - recall:+.6f})")

    results.sort(key=lambda item: item["importance_recall_drop"], reverse=True)
    report = {
        "experiment": "build5_knn_leave_one_group_out_importance",
        "interpretation": "Classifica del segnale predittivo KNN; non importanza dei pesi della rete neurale.",
        "selection_metric": "importance_recall_drop",
        "split_used_for_ranking": "validation",
        "knn": {
            "n_neighbors": args.n_neighbors,
            "weights": "distance",
            "train_examples": int(len(train_rows)),
            "validation_examples": int(len(validation_rows)),
            "projection_dimensions_for_dense_groups": args.projection_dimensions,
            "random_state": RANDOM_STATE,
        },
        "full_representation": {
            "validation_recall_at_6": full_recall,
            "feature_dimensions": full_dimensions,
            "elapsed_seconds": full_seconds,
        },
        "results_ranked_by_importance": results,
    }
    report_path = args.output_dir / "reports" / "build5_knn_feature_importance.json"
    write_json(report_path, report)
    print("\nClassifica KNN Build 5 completata:")
    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['removed_group']}: perdita Recall@6 {result['importance_recall_drop']:+.6f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
