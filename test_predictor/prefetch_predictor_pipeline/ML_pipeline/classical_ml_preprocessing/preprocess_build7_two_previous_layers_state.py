#!/usr/bin/env python3
"""Materialize Build 7 tabular state for layers l, l-1 and l-2.

This script only adds the third *state* block.  It does not create another
4096D hidden branch or another full 512D KV branch: those are separate FFN
inputs and are intentionally outside this Build 7 state experiment.
"""

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
PREFIXES = ("", "previous_", "previous_previous_")


def columns(prefix: str) -> tuple[list[str], list[str], list[str], list[str]]:
    return (
        [f"{prefix}routing_history_expert_{index}" for index in range(TOP_K)],
        [f"{prefix}router_logit_expert_id_{index}" for index in range(16)],
        [f"{prefix}router_logit_value_{index}" for index in range(16)],
        [f"{prefix}kv_current_{name}" for name in KV_STATS],
    )


def multihot(table: pd.DataFrame, names: list[str]) -> sparse.csr_matrix:
    ids = table[names].to_numpy(dtype=np.int16)
    if np.any(ids < 0) or np.any(ids >= NUM_EXPERTS):
        raise ValueError("Routing history fuori intervallo 0..255.")
    if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError("Routing history con expert duplicati.")
    rows = np.repeat(np.arange(len(table), dtype=np.int32), TOP_K)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, ids.reshape(-1))),
        shape=(len(table), NUM_EXPERTS),
        dtype=np.float32,
    )


def fixed_router_encoder(names: list[str]) -> OneHotEncoder:
    return OneHotEncoder(
        categories=[np.arange(NUM_EXPERTS) for _ in names],
        handle_unknown="error",
        sparse_output=True,
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea lo state CSR Build 7 con l, l-1 e l-2.")
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    parquet_files = sorted(args.training_dir.glob("prefetch_training*.parquet"))
    if len(parquet_files) != 1:
        raise ValueError("--training-dir deve contenere esattamente un prefetch_training*.parquet.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output gia' esistente: {args.output_dir}")

    groups = {prefix: columns(prefix) for prefix in PREFIXES}
    current_history, current_ids, current_values, current_kv = groups[""]
    required = [
        "split", "layer", "previous_layer", "previous_previous_layer",
        *[name for group in groups.values() for names in group for name in names],
    ]
    print("1) Carico lo state esteso Build 7 dal Parquet...")
    table = pd.read_parquet(parquet_files[0], columns=required)
    if set(table["split"].unique()) != set(SPLITS):
        raise ValueError("Split Build 7 mancanti o non validi.")
    if not np.all(table["previous_layer"].to_numpy() == table["layer"].to_numpy() - 1):
        raise ValueError("previous_layer non coincide con layer-1.")
    if not np.all(table["previous_previous_layer"].to_numpy() == table["layer"].to_numpy() - 2):
        raise ValueError("previous_previous_layer non coincide con layer-2.")
    split_tables = {split: table.loc[table["split"] == split].reset_index(drop=True) for split in SPLITS}

    layer_encoder = OneHotEncoder(handle_unknown="error", sparse_output=True, dtype=np.float32)
    layer_encoder.fit(split_tables["train"][["layer"]])
    id_encoders = {prefix: fixed_router_encoder(groups[prefix][1]) for prefix in PREFIXES}
    value_scalers = {prefix: StandardScaler().fit(split_tables["train"][groups[prefix][2]]) for prefix in PREFIXES}
    kv_scalers = {prefix: StandardScaler().fit(split_tables["train"][groups[prefix][3]]) for prefix in PREFIXES}
    for prefix in PREFIXES:
        id_encoders[prefix].fit(split_tables["train"][groups[prefix][1]])

    print("2) Materializzo CSR state corrente, precedente e pre-precedente...")
    matrices: dict[str, sparse.csr_matrix] = {}
    labels: dict[str, np.ndarray] = {}
    for split in SPLITS:
        frame = split_tables[split]
        y = np.load(args.training_dir / "arrays" / f"y_{split}.npy")
        if y.shape != (len(frame), NUM_EXPERTS) or y.dtype != np.uint8 or not np.all(y.sum(axis=1) == TOP_K):
            raise ValueError(f"Label incompatibili per {split}.")
        parts: list[sparse.csr_matrix] = [
            multihot(frame, current_history),
            layer_encoder.transform(frame[["layer"]]).tocsr(),
            id_encoders[""].transform(frame[current_ids]).tocsr(),
            sparse.csr_matrix(value_scalers[""].transform(frame[current_values]).astype(np.float32)),
            sparse.csr_matrix(kv_scalers[""].transform(frame[current_kv]).astype(np.float32)),
        ]
        for prefix in ("previous_", "previous_previous_"):
            history, ids, values, kv = groups[prefix]
            parts.extend([
                multihot(frame, history),
                id_encoders[prefix].transform(frame[ids]).tocsr(),
                sparse.csr_matrix(value_scalers[prefix].transform(frame[values]).astype(np.float32)),
                sparse.csr_matrix(kv_scalers[prefix].transform(frame[kv]).astype(np.float32)),
            ])
        matrices[split] = sparse.hstack(parts, format="csr", dtype=np.float32)
        labels[split] = y

    arrays = args.output_dir / "arrays"
    artifacts = args.output_dir / "artifacts"
    arrays.mkdir(parents=True)
    artifacts.mkdir()
    for split in SPLITS:
        sparse.save_npz(arrays / f"X_classic_{split}.npz", matrices[split])
        np.save(arrays / f"y_{split}.npy", labels[split])

    feature_names = [
        *[f"routing_history_multihot_expert_{index}" for index in range(NUM_EXPERTS)],
        *layer_encoder.get_feature_names_out(["layer"]).tolist(),
        *id_encoders[""].get_feature_names_out(current_ids).tolist(), *current_values, *current_kv,
    ]
    for prefix in ("previous_", "previous_previous_"):
        history, ids, values, kv = groups[prefix]
        feature_names.extend([
            *[f"{prefix}routing_history_multihot_expert_{index}" for index in range(NUM_EXPERTS)],
            *id_encoders[prefix].get_feature_names_out(ids).tolist(), *values, *kv,
        ])
    (artifacts / "feature_names.json").write_text(json.dumps({"feature_names": feature_names}, indent=2) + "\n", encoding="utf-8")
    artifacts_to_save = {"layer_encoder.joblib": layer_encoder}
    for prefix in PREFIXES:
        label = prefix or "current_"
        artifacts_to_save[f"{label}router_id_encoder.joblib"] = id_encoders[prefix]
        artifacts_to_save[f"{label}router_values_scaler.joblib"] = value_scalers[prefix]
        artifacts_to_save[f"{label}kv_statistics_scaler.joblib"] = kv_scalers[prefix]
    for name, value in artifacts_to_save.items():
        joblib.dump(value, artifacts / name)
    (artifacts / "preprocessing_summary.json").write_text(
        json.dumps(
            {
                "experiment": "build7_current_previous_previous_state",
                "feature_count": len(feature_names),
                "target": "E_l(t+1)",
                "target_layers": [5, 42],
                "scalers_fit_split": "train",
                "splits": {split: list(matrices[split].shape) for split in SPLITS},
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"State Build 7 completato: {len(feature_names)} feature -> {args.output_dir}")


if __name__ == "__main__":
    main()
