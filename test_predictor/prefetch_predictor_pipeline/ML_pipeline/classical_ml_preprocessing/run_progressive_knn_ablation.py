#!/usr/bin/env python3
"""Screen semantic feature groups with the same KNN configuration."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_classical_models import (
    TOP_K,
    calcola_recall_at_6,
    estrai_score_positivi,
    seleziona_train_knn,
)


SPLITS = ("train", "validation", "test")
NUM_EXPERTS = 256
MIN_FREQUENCY = 41
RANDOM_STATE = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_TRAINING_DIR = PIPELINE_ROOT / "build_dataset_2" / "ml" / "training"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "build_dataset_2" / "ml" / "ablations" / "progressive_knn"

ROUTING_COLUMNS = [f"routing_history_expert_{index}" for index in range(TOP_K)]
ROUTER_ID_COLUMNS = [f"router_logit_expert_id_{index}" for index in range(16)]
ROUTER_VALUE_COLUMNS = [f"router_logit_value_{index}" for index in range(16)]


# Every configuration always contains the same H=1 routing multi-hot vector.
# The remaining groups are added one source of information at a time.
CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "01_routing_multihot",
        "description": "Solo l'insieme multi-hot degli expert in E_l(t).",
        "categorical_columns": [],
        "numeric_columns": [],
    },
    {
        "name": "02_routing_multihot_layer",
        "description": "Routing multi-hot piu' layer routed corrente.",
        "categorical_columns": ["layer"],
        "numeric_columns": [],
    },
    {
        "name": "03_routing_layer_router_state",
        "description": "Routing, layer, ID e valori dei 16 raw router logits.",
        "categorical_columns": ["layer", *ROUTER_ID_COLUMNS],
        "numeric_columns": [*ROUTER_VALUE_COLUMNS],
    },
    {
        "name": "04_routing_layer_token_context",
        "description": "Routing, layer, token_id e token_position.",
        "categorical_columns": ["layer", "token_id"],
        "numeric_columns": ["token_position"],
    },
    {
        "name": "05_routing_layer_hidden_pca",
        "description": "Routing, layer e le 64 coordinate PCA di ffn_norm.",
        "categorical_columns": ["layer"],
        "numeric_columns": [f"hidden_pca_{index}" for index in range(64)],
    },
    {
        "name": "06_routing_layer_router_state_hidden_pca",
        "description": "Routing, layer, router state e hidden PCA-64.",
        "categorical_columns": ["layer", *ROUTER_ID_COLUMNS],
        "numeric_columns": [*ROUTER_VALUE_COLUMNS, *[f"hidden_pca_{index}" for index in range(64)]],
    },
    {
        "name": "07_all_available_with_multihot",
        "description": "Tutte le feature classiche disponibili, con routing multi-hot.",
        "categorical_columns": ["layer", "token_id", *ROUTER_ID_COLUMNS],
        "numeric_columns": ["token_position", *ROUTER_VALUE_COLUMNS, *[f"hidden_pca_{index}" for index in range(64)]],
    },
)


def leggi_argomenti() -> argparse.Namespace:
    """
    Read paths, selected configurations and the fixed KNN train-sample limit.

    Args:
        No Python arguments. Values are supplied with ``--training-dir``,
        ``--output-dir``, ``--configurations`` and ``--knn-max-train-samples``.

    Returns:
        Namespace containing the screening configuration.
    """
    parser = argparse.ArgumentParser(
        description="Esegue ablation progressiva di gruppi di feature con KNN."
    )
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--configurations",
        default="all",
        help="all oppure nomi separati da virgole delle configurazioni da eseguire.",
    )
    parser.add_argument(
        "--knn-max-train-samples",
        type=int,
        default=20_000,
        help="Massimo numero di righe train per ogni KNN. Default: 20000.",
    )
    return parser.parse_args()


def trova_parquet_base(training_dir: Path) -> Path:
    """
    Find the single base Parquet table inside one build training directory.

    Args:
        training_dir: Directory containing one ``prefetch_training`` Parquet file.

    Returns:
        Path of the compatible Parquet table.

    Raises:
        FileNotFoundError: If no compatible Parquet table exists.
        ValueError: If more than one compatible Parquet table exists.
    """
    candidates = sorted(training_dir.glob("prefetch_training*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"Nessun Parquet prefetch_training*.parquet trovato in: {training_dir}"
        )
    if len(candidates) != 1:
        raise ValueError(
            "La training directory deve contenere un solo Parquet compatibile: "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def valida_colonne(table: pd.DataFrame) -> None:
    """
    Validate the columns required by every configured feature group.

    Args:
        table: Full BUILD_DATASET_2 training table.

    Returns:
        None. Raises an error before any KNN is trained when input is incomplete.

    Raises:
        ValueError: If a split or a required feature column is missing.
    """
    required_columns = {"split", *ROUTING_COLUMNS}
    for configuration in CONFIGURATIONS:
        required_columns.update(configuration["categorical_columns"])
        required_columns.update(configuration["numeric_columns"])
    missing_columns = sorted(required_columns - set(table.columns))
    if missing_columns:
        raise ValueError(f"Nel Parquet mancano queste colonne: {missing_columns}")
    found_splits = set(table["split"].unique())
    if found_splits != set(SPLITS):
        raise ValueError(f"Split non validi: trovati {sorted(found_splits)}")


def carica_label(training_dir: Path, split: str, rows: int) -> np.ndarray:
    """
    Load and validate the multi-label target for one split.

    Args:
        training_dir: Directory containing ``arrays/y_<split>.npy``.
        split: Split name: train, validation or test.
        rows: Expected number of examples in the split.

    Returns:
        Unsigned 8-bit target matrix with shape ``[rows, 256]``.

    Raises:
        FileNotFoundError: If the expected target file does not exist.
        ValueError: If shape or active-label count is invalid.
    """
    path = training_dir / "arrays" / f"y_{split}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Label non trovate: {path}")
    labels = np.load(path)
    if labels.shape != (rows, NUM_EXPERTS):
        raise ValueError(f"{split}: y ha shape {labels.shape}, attesa {(rows, NUM_EXPERTS)}")
    if labels.dtype != np.uint8 or not np.all(labels.sum(axis=1) == TOP_K):
        raise ValueError(f"{split}: ogni label deve essere uint8 con {TOP_K} expert attivi")
    return labels


def crea_multihot(table: pd.DataFrame) -> sparse.csr_matrix:
    """
    Build the order-independent H=1 routing vector with 256 expert positions.

    Args:
        table: Rows of one split with the six routing-history IDs.

    Returns:
        CSR float32 matrix with shape ``[N, 256]`` and six ones per row.

    Raises:
        ValueError: If IDs are invalid or an input routing contains duplicates.
    """
    expert_ids = table[ROUTING_COLUMNS].to_numpy(dtype=np.int16)
    if np.any(expert_ids < 0) or np.any(expert_ids >= NUM_EXPERTS):
        raise ValueError("Gli ID della routing history devono essere compresi tra 0 e 255.")
    if np.any(np.diff(np.sort(expert_ids, axis=1), axis=1) == 0):
        raise ValueError("Ogni routing history deve contenere sei expert distinti.")

    rows = np.repeat(np.arange(len(table), dtype=np.int32), TOP_K)
    columns = expert_ids.reshape(-1).astype(np.int32)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(table), NUM_EXPERTS),
        dtype=np.float32,
    )
    if not np.all(np.diff(matrix.indptr) == TOP_K):
        raise ValueError("Ogni riga multi-hot deve avere esattamente sei valori attivi.")
    return matrix


def trasforma_configurazione(
    configuration: dict[str, Any],
    split_tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, sparse.csr_matrix], dict[str, Any]]:
    """
    Fit train-only transformers and build the feature matrices for one group set.

    Args:
        configuration: One item from ``CONFIGURATIONS``.
        split_tables: Mapping containing train, validation and test DataFrames.

    Returns:
        Tuple with CSR matrices by split and serializable transformation metadata.
    """
    categorical_columns: list[str] = configuration["categorical_columns"]
    numeric_columns: list[str] = configuration["numeric_columns"]

    encoder: OneHotEncoder | None = None
    scaler: StandardScaler | None = None
    if categorical_columns:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=MIN_FREQUENCY,
            sparse_output=True,
            dtype=np.float32,
        )
        encoder.fit(split_tables["train"][categorical_columns])
    if numeric_columns:
        scaler = StandardScaler()
        scaler.fit(split_tables["train"][numeric_columns])

    matrices: dict[str, sparse.csr_matrix] = {}
    for split in SPLITS:
        parts: list[sparse.csr_matrix] = [crea_multihot(split_tables[split])]
        if encoder is not None:
            parts.append(encoder.transform(split_tables[split][categorical_columns]).tocsr())
        if scaler is not None:
            numeric = scaler.transform(split_tables[split][numeric_columns]).astype(np.float32)
            parts.append(sparse.csr_matrix(numeric))
        matrices[split] = sparse.hstack(parts, format="csr", dtype=np.float32)

    metadata: dict[str, Any] = {
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "routing_representation": "multihot_256",
        "one_hot_configuration": None,
    }
    if encoder is not None:
        metadata["one_hot_configuration"] = {
            "handle_unknown": "infrequent_if_exist",
            "min_frequency": MIN_FREQUENCY,
            "one_hot_feature_count": len(encoder.get_feature_names_out()),
        }
    return matrices, metadata


def scegli_configurazioni(selection: str) -> list[dict[str, Any]]:
    """
    Select all named configurations or a comma-separated subset.

    Args:
        selection: ``all`` or comma-separated configuration names.

    Returns:
        Ordered list of configurations to evaluate.

    Raises:
        ValueError: If a requested configuration name does not exist.
    """
    by_name = {configuration["name"]: configuration for configuration in CONFIGURATIONS}
    if selection == "all":
        return list(CONFIGURATIONS)
    names = [name.strip() for name in selection.split(",") if name.strip()]
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"Configurazioni non disponibili: {unknown}")
    return [by_name[name] for name in names]


def salva_json(path: Path, content: dict[str, Any]) -> None:
    """
    Write a readable JSON artifact, creating its parent directory if needed.

    Args:
        path: Destination JSON file.
        content: Serializable dictionary to write.

    Returns:
        None. The JSON is written with two-space indentation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """
    Run the controlled KNN feature-group screening on BUILD_DATASET_2.

    Args:
        No Python arguments. Paths, configuration subset and the KNN train cap
        are read by ``leggi_argomenti()``.

    Returns:
        None. Saves configuration and metric JSON files; it does not save KNN
        models or persistent intermediate matrices.
    """
    args = leggi_argomenti()
    configurations = scegli_configurazioni(args.configurations)
    if args.knn_max_train_samples < 1:
        raise ValueError("--knn-max-train-samples deve essere un intero positivo.")

    print("1) Carico Parquet, split e label BUILD_DATASET_2...")
    table = pd.read_parquet(trova_parquet_base(args.training_dir))
    valida_colonne(table)
    split_tables = {
        split: table.loc[table["split"] == split].reset_index(drop=True)
        for split in SPLITS
    }
    labels = {
        split: carica_label(args.training_dir, split, len(split_tables[split]))
        for split in SPLITS
    }

    artifacts_dir = args.output_dir / "artifacts"
    reports_dir = args.output_dir / "reports"
    salva_json(
        artifacts_dir / "ablation_configurations.json",
        {
            "experiment": "progressive_knn_feature_ablation",
            "source_training_dir": str(args.training_dir),
            "target": "E_l(t+1)",
            "routing_base": "E_l(t) encoded as multihot_256",
            "knn": {
                "n_neighbors": 5,
                "weights": "distance",
                "n_jobs": -1,
                "random_state_for_train_subset": RANDOM_STATE,
                "max_train_samples": args.knn_max_train_samples,
            },
            "configurations": configurations,
        },
    )

    print("2) Valuto le configurazioni KNN...")
    results: list[dict[str, Any]] = []
    for index, configuration in enumerate(configurations, start=1):
        print(f"   [{index}/{len(configurations)}] {configuration['name']}...")
        matrices, metadata = trasforma_configurazione(configuration, split_tables)
        x_fit, y_fit = seleziona_train_knn(
            matrices["train"], labels["train"], args.knn_max_train_samples
        )
        model = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        start = time.perf_counter()
        model.fit(x_fit, y_fit)
        train_scores = estrai_score_positivi(model, matrices["train"])
        validation_scores = estrai_score_positivi(model, matrices["validation"])
        elapsed_seconds = time.perf_counter() - start
        result = {
            "name": configuration["name"],
            "description": configuration["description"],
            "feature_count": int(matrices["train"].shape[1]),
            "train_examples_used": int(x_fit.shape[0]),
            "train_recall_at_6": calcola_recall_at_6(train_scores, labels["train"]),
            "validation_recall_at_6": calcola_recall_at_6(
                validation_scores, labels["validation"]
            ),
            "elapsed_seconds": elapsed_seconds,
            "matrices": {
                split: {"shape": list(matrices[split].shape), "nnz": int(matrices[split].nnz)}
                for split in ("train", "validation")
            },
            "transformation": metadata,
        }
        results.append(result)
        print(
            f"      validation Recall@6: {result['validation_recall_at_6']:.4f} "
            f"({result['feature_count']} feature)"
        )

    results.sort(key=lambda result: result["validation_recall_at_6"], reverse=True)
    summary = {
        "experiment": "progressive_knn_feature_ablation",
        "metric_for_selection": "validation_recall_at_6",
        "test_evaluated": False,
        "results_ranked_by_validation": results,
    }
    salva_json(reports_dir / "progressive_knn_summary.json", summary)

    print("\nAblation KNN completata.")
    for result in results:
        print(f"{result['name']}: validation Recall@6 = {result['validation_recall_at_6']:.4f}")
    print(f"Report: {reports_dir / 'progressive_knn_summary.json'}")


if __name__ == "__main__":
    main()
