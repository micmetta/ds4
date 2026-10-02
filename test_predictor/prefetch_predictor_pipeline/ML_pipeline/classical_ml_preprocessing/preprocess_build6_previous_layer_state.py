#!/usr/bin/env python3
"""Materialize the Build 6 current plus preceding-layer state as CSR arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SPLITS = ("train", "validation", "test")
TOP_K = 6
NUM_EXPERTS = 256
KV_STATS = ("l2_norm", "rms", "mean", "std", "max_abs")


def columns(prefix: str) -> tuple[list[str], list[str], list[str]]:
    return (
        [f"{prefix}routing_history_expert_{index}" for index in range(TOP_K)],
        [f"{prefix}router_logit_expert_id_{index}" for index in range(16)],
        [f"{prefix}router_logit_value_{index}" for index in range(16)],
    )


CURRENT_HISTORY, CURRENT_IDS, CURRENT_VALUES = columns("")
PREVIOUS_HISTORY, PREVIOUS_IDS, PREVIOUS_VALUES = columns("previous_")
CURRENT_KV = [f"kv_current_{name}" for name in KV_STATS]
PREVIOUS_KV = [f"previous_kv_current_{name}" for name in KV_STATS]


def multihot(table: pd.DataFrame, names: list[str]) -> sparse.csr_matrix:
    ids = table[names].to_numpy(dtype=np.int16)
    if np.any(ids < 0) or np.any(ids >= NUM_EXPERTS):
        raise ValueError("Routing history fuori intervallo 0..255.")
    if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError("Routing history con expert duplicati.")
    rows = np.repeat(np.arange(len(table), dtype=np.int32), TOP_K)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, ids.reshape(-1))), shape=(len(table), NUM_EXPERTS), dtype=np.float32)


def fixed_router_encoder(names: list[str]) -> OneHotEncoder:
    return OneHotEncoder(categories=[np.arange(NUM_EXPERTS) for _ in names], handle_unknown="error", sparse_output=True, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea lo state CSR Build 6 con layer precedente.")
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    parquet_files = sorted(args.training_dir.glob("prefetch_training*.parquet"))
    if len(parquet_files) != 1:
        raise ValueError("--training-dir deve contenere esattamente un prefetch_training*.parquet.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output gia' esistente: {args.output_dir}")

    required = ["split", "layer", "previous_layer", *CURRENT_HISTORY, *CURRENT_IDS, *CURRENT_VALUES, *CURRENT_KV, *PREVIOUS_HISTORY, *PREVIOUS_IDS, *PREVIOUS_VALUES, *PREVIOUS_KV]
    print("1) Carico lo state esteso Build 6 dal Parquet...")
    table = pd.read_parquet(parquet_files[0], columns=required)
    if set(table["split"].unique()) != set(SPLITS):
        raise ValueError("Split Build 6 mancanti o non validi.")
    if not np.all(table["previous_layer"].to_numpy() == table["layer"].to_numpy() - 1):
        raise ValueError("previous_layer non coincide con layer-1.")
    split_tables = {split: table.loc[table["split"] == split].reset_index(drop=True) for split in SPLITS}

    layer_encoder = OneHotEncoder(handle_unknown="error", sparse_output=True, dtype=np.float32)
    layer_encoder.fit(split_tables["train"][["layer"]])
    current_id_encoder = fixed_router_encoder(CURRENT_IDS)
    previous_id_encoder = fixed_router_encoder(PREVIOUS_IDS)
    current_id_encoder.fit(split_tables["train"][CURRENT_IDS])
    previous_id_encoder.fit(split_tables["train"][PREVIOUS_IDS])
    current_values_scaler = StandardScaler().fit(split_tables["train"][CURRENT_VALUES])
    previous_values_scaler = StandardScaler().fit(split_tables["train"][PREVIOUS_VALUES])
    current_kv_scaler = StandardScaler().fit(split_tables["train"][CURRENT_KV])
    previous_kv_scaler = StandardScaler().fit(split_tables["train"][PREVIOUS_KV])

    print("2) Materializzo CSR state corrente + precedente e verifico le label...")
    matrices: dict[str, sparse.csr_matrix] = {}
    labels: dict[str, np.ndarray] = {}
    for split in SPLITS:
        table_split = split_tables[split]
        y = np.load(args.training_dir / "arrays" / f"y_{split}.npy")
        if y.shape != (len(table_split), NUM_EXPERTS) or y.dtype != np.uint8 or not np.all(y.sum(axis=1) == TOP_K):
            raise ValueError(f"Label incompatibili per {split}.")
        matrices[split] = sparse.hstack([
            multihot(table_split, CURRENT_HISTORY), layer_encoder.transform(table_split[["layer"]]).tocsr(),
            current_id_encoder.transform(table_split[CURRENT_IDS]).tocsr(),
            sparse.csr_matrix(current_values_scaler.transform(table_split[CURRENT_VALUES]).astype(np.float32)),
            sparse.csr_matrix(current_kv_scaler.transform(table_split[CURRENT_KV]).astype(np.float32)),
            multihot(table_split, PREVIOUS_HISTORY), previous_id_encoder.transform(table_split[PREVIOUS_IDS]).tocsr(),
            sparse.csr_matrix(previous_values_scaler.transform(table_split[PREVIOUS_VALUES]).astype(np.float32)),
            sparse.csr_matrix(previous_kv_scaler.transform(table_split[PREVIOUS_KV]).astype(np.float32)),
        ], format="csr", dtype=np.float32)
        labels[split] = y

    arrays = args.output_dir / "arrays"
    artifacts = args.output_dir / "artifacts"
    arrays.mkdir(parents=True)
    artifacts.mkdir()
    for split in SPLITS:
        sparse.save_npz(arrays / f"X_classic_{split}.npz", matrices[split])
        np.save(arrays / f"y_{split}.npy", labels[split])
    feature_names = [
        *[f"routing_history_multihot_expert_{index}" for index in range(NUM_EXPERTS)], *layer_encoder.get_feature_names_out(["layer"]).tolist(),
        *current_id_encoder.get_feature_names_out(CURRENT_IDS).tolist(), *CURRENT_VALUES, *CURRENT_KV,
        *[f"previous_routing_history_multihot_expert_{index}" for index in range(NUM_EXPERTS)],
        *previous_id_encoder.get_feature_names_out(PREVIOUS_IDS).tolist(), *PREVIOUS_VALUES, *PREVIOUS_KV,
    ]
    (artifacts / "feature_names.json").write_text(json.dumps({"feature_names": feature_names}, indent=2) + "\n", encoding="utf-8")
    for name, value in {
        "layer_encoder.joblib": layer_encoder, "current_router_id_encoder.joblib": current_id_encoder,
        "previous_router_id_encoder.joblib": previous_id_encoder, "current_router_values_scaler.joblib": current_values_scaler,
        "previous_router_values_scaler.joblib": previous_values_scaler, "current_kv_statistics_scaler.joblib": current_kv_scaler,
        "previous_kv_statistics_scaler.joblib": previous_kv_scaler,
    }.items():
        joblib.dump(value, artifacts / name)
    (artifacts / "preprocessing_summary.json").write_text(json.dumps({
        "experiment": "build6_current_and_previous_layer_state", "feature_count": len(feature_names), "target": "E_l(t+1)",
        "target_layers": [4, 42], "scalers_fit_split": "train", "splits": {split: list(matrices[split].shape) for split in SPLITS},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"State Build 6 completato: {len(feature_names)} feature -> {args.output_dir}")


if __name__ == "__main__":
    main()
