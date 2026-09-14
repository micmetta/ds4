#!/usr/bin/env python3
"""Persistent CPU service used only by TEST_7 learned-expert prefetching.

DS4 sends one binary request for every routed ``(token t, layer l)``.  This
process reproduces Build-5 preprocessing, evaluates the selected Keras
checkpoint on the CPU, then returns six expert IDs predicted for ``t + 1``.
It deliberately owns TensorFlow: DS4 stays a C/CUDA inference executable.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

PIPELINE_NEURAL_NETWORK = (
    Path(__file__).resolve().parents[1]
    / "prefetch_predictor_pipeline"
    / "ML_pipeline"
    / "neural_network"
)
sys.path.insert(0, str(PIPELINE_NEURAL_NETWORK))

from ffn_utils import (  # noqa: E402
    build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn,
    require_tensorflow,
)


EXPERTS = 256
TOP_K = 6
ROUTER_LOGITS = 16
HIDDEN_DIMENSION = 4096
KV_DIMENSION = 512
ROUTED_LAYER_FIRST = 3
REQUEST = struct.Struct("<4sIIII6i256f4096f512f")
RESPONSE = struct.Struct("<4sI6id")


def receive_exact(connection: socket.socket, size: int) -> bytes:
    """Read exactly one complete fixed-size request from the local socket."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class RuntimePredictor:
    """Loads frozen Build-5 artifacts once and converts one DS4 event at a time."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.layer_encoder = joblib.load(args.state_artifacts / "categorical_encoder.joblib")
        self.router_scaler = joblib.load(args.state_artifacts / "numeric_scaler.joblib")
        self.router_id_encoder = joblib.load(args.state_artifacts / "router_id_encoder.joblib")
        self.kv_stats_scaler = joblib.load(args.state_artifacts / "kv_current_scaler.joblib")
        self.hidden_means = np.load(args.hidden_artifacts / "hidden_train_mean_by_layer.npy")
        self.hidden_stds = np.load(args.hidden_artifacts / "hidden_train_std_by_layer.npy")
        self.kv_means = np.load(args.kv_artifacts / "kv_train_mean_by_layer.npy")
        self.kv_stds = np.load(args.kv_artifacts / "kv_train_std_by_layer.npy")
        self.token_ids = np.load(args.token_embedding_dir / "token_ids.npy", mmap_mode="r")
        self.token_embeddings = np.load(
            args.token_embedding_dir / "token_embeddings_f16.npy", mmap_mode="r"
        )
        token_summary = __import__("json").loads(
            (args.token_data_dir / "artifacts" / "preprocessing_summary.json").read_text()
        )
        token_position = token_summary["token_position"]
        self.position_mean = float(token_position["train_mean"])
        self.position_std = float(token_position["train_std"])
        self.hidden_scale = float(args.hidden_scale)

        tf = require_tensorflow()
        # TEST_7 must not contend with DS4 for the CUDA GPU. TensorFlow is
        # therefore made CPU-only before any model or tensor is created.
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as error:
            raise RuntimeError("TensorFlow ha inizializzato la GPU prima del servizio TEST_7") from error
        # The winning Hyperband trial and its test evaluation used float32.
        tf.keras.mixed_precision.set_global_policy("float32")
        self.model = build_fixed_nnz_state_raw_hidden_token_embedding_kv_ffn(
            state_feature_count=4413,
            state_input_slots=45,
            frozen_token_embeddings=self.token_embeddings,
            state_hidden_units=512,
            fusion_hidden_layers=[512],
            activation="relu",
            dropout=0.3,
            learning_rate=0.00015,
            l2_regularization=0.0,
            tf=tf,
            hidden_projection_dimension=512,
            token_projection_dimension=64,
            raw_hidden_projection_activation="relu",
            token_projection_activation="relu",
            kv_projection_dimension=None,
            kv_projection_activation="relu",
            raw_hidden_dequantization_scale=self.hidden_scale,
            # La CPU di questa macchina non accelera float16. Il checkpoint
            # contiene gli stessi pesi: cambiamo solo il tipo dei calcoli
            # runtime per evitare il fallback CPU estremamente lento.
            raw_hidden_dequantization_dtype="float32",
            raw_hidden_projection_dtype="float32",
            kv_input_dtype="float32",
        )
        self.model.load_weights(args.checkpoint)

    def _compact_token_index(self, token_id: int) -> int:
        index = int(np.searchsorted(self.token_ids, token_id))
        if index >= len(self.token_ids) or int(self.token_ids[index]) != token_id:
            raise ValueError(f"token_id {token_id} non presente nel lookup Build 5")
        return index

    def _state_inputs(
        self, layer: int, selected: np.ndarray, ranked_ids: np.ndarray,
        ranked_logits: np.ndarray, kv: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild exactly the 44 non-zero state features plus the bias sentinel."""
        values = {
            "layer": [layer],
            **{f"router_logit_expert_id_{i}": [int(ranked_ids[i])] for i in range(ROUTER_LOGITS)},
            **{f"router_logit_value_{i}": [float(ranked_logits[i])] for i in range(ROUTER_LOGITS)},
        }
        row = pd.DataFrame(values)
        history_rows = sparse.csr_matrix(
            (np.ones(TOP_K, dtype=np.float32), (np.zeros(TOP_K, dtype=np.int32), selected)),
            shape=(1, EXPERTS), dtype=np.float32,
        )
        layer_rows = self.layer_encoder.transform(row[["layer"]]).tocsr()
        id_columns = [f"router_logit_expert_id_{i}" for i in range(ROUTER_LOGITS)]
        logit_columns = [f"router_logit_value_{i}" for i in range(ROUTER_LOGITS)]
        router_id_rows = self.router_id_encoder.transform(row[id_columns]).tocsr()
        logits_rows = sparse.csr_matrix(
            self.router_scaler.transform(row[logit_columns]).astype(np.float32)
        )
        kv_statistics = np.asarray([[np.linalg.norm(kv), np.sqrt(np.mean(np.square(kv))),
                                     np.mean(kv), np.std(kv), np.max(np.abs(kv))]], dtype=np.float32)
        kv_rows = sparse.csr_matrix(
            self.kv_stats_scaler.transform(kv_statistics).astype(np.float32)
        )
        state = sparse.hstack(
            [history_rows, layer_rows, router_id_rows, logits_rows, kv_rows],
            format="csr", dtype=np.float32,
        )
        if state.shape != (1, 4413) or state.nnz != 44:
            raise ValueError(f"state runtime inatteso: shape={state.shape}, nnz={state.nnz}")
        indices = np.empty((1, 45), dtype=np.int32)
        feature_values = np.empty((1, 45), dtype=np.float32)
        indices[0, :44] = state.indices
        feature_values[0, :44] = state.data
        indices[0, 44] = 4413
        feature_values[0, 44] = 1.0
        return indices, feature_values

    def predict(
        self, pos: int, token: int, layer: int, selected: np.ndarray,
        router_logits: np.ndarray, raw_hidden: np.ndarray, kv: np.ndarray,
    ) -> np.ndarray:
        """Apply the saved Build-5 transformations and return the six best experts."""
        layer_index = layer - ROUTED_LAYER_FIRST
        if layer_index < 0 or layer_index >= len(self.hidden_means):
            raise ValueError(f"layer routed non valido: {layer}")
        ranked_ids = np.argsort(-router_logits, kind="stable")[:ROUTER_LOGITS].astype(np.int32)
        ranked_logits = router_logits[ranked_ids]
        state_indices, state_values = self._state_inputs(
            layer, selected, ranked_ids, ranked_logits, kv
        )
        normalized_hidden = (raw_hidden - self.hidden_means[layer_index]) / self.hidden_stds[layer_index]
        hidden_int8 = np.clip(
            np.rint(normalized_hidden / self.hidden_scale), -127, 127
        ).astype(np.int8)[None, :]
        normalized_kv = ((kv - self.kv_means[layer_index]) / self.kv_stds[layer_index]).astype(np.float32)[None, :]
        inputs = {
            "state_feature_indices": state_indices,
            "state_feature_values": state_values,
            "raw_hidden_state": hidden_int8,
            "kv_current_state": normalized_kv,
            "token_embedding_index": np.asarray([self._compact_token_index(token)], dtype=np.int32),
            "token_position": np.asarray([[(pos - self.position_mean) / self.position_std]], dtype=np.float32),
        }
        scores = self.model(inputs, training=False).numpy()[0]
        # Stable descending order makes tied scores deterministic.
        return np.argsort(-scores, kind="stable")[:TOP_K].astype(np.int32)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Servizio CPU del predictor Build 5 per TEST_7.")
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-artifacts", type=Path, required=True)
    parser.add_argument("--hidden-artifacts", type=Path, required=True)
    parser.add_argument("--kv-artifacts", type=Path, required=True)
    parser.add_argument("--token-data-dir", type=Path, required=True)
    parser.add_argument("--token-embedding-dir", type=Path, required=True)
    parser.add_argument("--hidden-scale", type=float, default=0.03066560039370079)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.socket_path.exists():
        args.socket_path.unlink()
    print("TEST_7: carico checkpoint e artefatti Build 5 sulla CPU...", flush=True)
    predictor = RuntimePredictor(args)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket_path))
    server.listen(1)
    print(f"TEST_7: predictor pronto su {args.socket_path}", flush=True)
    try:
        while True:
            connection, _ = server.accept()
            with connection:
                while True:
                    try:
                        payload = receive_exact(connection, REQUEST.size)
                    except EOFError:
                        break
                    values = REQUEST.unpack(payload)
                    magic, pos, token, layer, request_id = values[:5]
                    if magic != b"T7P1":
                        raise ValueError("protocollo TEST_7 non riconosciuto")
                    selected = np.asarray(values[5:11], dtype=np.int32)
                    logits_begin = 11
                    hidden_begin = logits_begin + EXPERTS
                    kv_begin = hidden_begin + HIDDEN_DIMENSION
                    logits = np.asarray(values[logits_begin:hidden_begin], dtype=np.float32)
                    hidden = np.asarray(values[hidden_begin:kv_begin], dtype=np.float32)
                    kv = np.asarray(values[kv_begin:kv_begin + KV_DIMENSION], dtype=np.float32)
                    started = time.perf_counter()
                    experts = predictor.predict(pos, token, layer, selected, logits, hidden, kv)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    connection.sendall(RESPONSE.pack(b"T7R1", request_id, *experts, elapsed_ms))
    finally:
        server.close()
        if args.socket_path.exists():
            args.socket_path.unlink()


if __name__ == "__main__":
    main()
