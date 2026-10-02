#!/usr/bin/env python3
"""CPU-only Build-7 Top-4 predictor used exclusively by TEST_14.

The service preserves TEST_13's asynchronous inference contract.  DS4 sends
the causal routing event at physical layer ``l`` plus raw hidden vectors for
``l`` and ``l-1``; this process recreates the fixed-NNZ current state and the
transition vector ``q`` before returning six experts for token ``t + 1``.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NEURAL_NETWORK = ROOT / "test_predictor" / "prefetch_predictor_pipeline" / "ML_pipeline" / "neural_network"
sys.path.insert(0, str(NEURAL_NETWORK))

from ffn_utils import build_fixed_nnz_state_two_hidden_gated_residual_ffn, require_tensorflow  # noqa: E402


EXPERTS = 256
TOP_K = 6
RANKED_ROUTER = 16
HIDDEN = 4096
KV = 512
FIRST_MODEL_LAYER = 5
LAST_MODEL_LAYER = 42
REQUEST = struct.Struct("<4sIIII6i256f4096f4096f512f")
RESPONSE = struct.Struct("<4sI6id")


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def load_keras_weights(keras_path: Path, model) -> None:
    """Load weights only, avoiding restoration of the unused Adam state."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        weights_path = Path(temporary_directory) / "model.weights.h5"
        with zipfile.ZipFile(keras_path) as archive:
            with archive.open("model.weights.h5") as source, weights_path.open("wb") as destination:
                destination.write(source.read())
        model.load_weights(weights_path)


class RuntimePredictor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.layer_encoder = joblib.load(args.state_artifacts / "layer_encoder.joblib")
        self.router_id_encoder = joblib.load(args.state_artifacts / "current_router_id_encoder.joblib")
        self.router_values_scaler = joblib.load(args.state_artifacts / "current_router_values_scaler.joblib")
        self.kv_scaler = joblib.load(args.state_artifacts / "current_kv_statistics_scaler.joblib")
        metadata = json.loads((args.fixed_nnz_artifacts / "fixed_nnz_state_metadata.json").read_text())
        expected_metadata = {
            "format": "fixed_nnz_state_slots_v1",
            "feature_count": 4411,
            "nonzero_features_per_example": 44,
            "input_slots": 45,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"Metadata fixed-NNZ Build 7 inattesi: {metadata}")
        self.current_means = np.load(args.current_hidden_artifacts / "hidden_train_mean_by_layer.npy")
        self.current_stds = np.load(args.current_hidden_artifacts / "hidden_train_std_by_layer.npy")
        self.previous_means = np.load(args.previous_hidden_artifacts / "hidden_train_mean_by_layer.npy")
        self.previous_stds = np.load(args.previous_hidden_artifacts / "hidden_train_std_by_layer.npy")
        self.transition = np.load(args.transition_artifacts / "transition_probabilities_train_global_float32.npy")
        if self.transition.shape != (40, EXPERTS, EXPERTS):
            raise ValueError(f"Matrice q inattesa: {self.transition.shape}")

        tf = require_tensorflow()
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as error:
            raise RuntimeError("TensorFlow ha inizializzato la GPU prima del servizio TEST_14") from error
        tf.keras.mixed_precision.set_global_policy("float32")
        self.model = build_fixed_nnz_state_two_hidden_gated_residual_ffn(
            4411, 45, 2048, 64, 64,
            args.current_hidden_scale, args.previous_hidden_scale,
            2048, 0.3, 1.5e-4, 3e-6, tf,
        )
        load_keras_weights(args.checkpoint, self.model)
        self.current_scale = float(args.current_hidden_scale)
        self.previous_scale = float(args.previous_hidden_scale)

    def _state_inputs(self, layer: int, selected: np.ndarray,
                      router_logits: np.ndarray, kv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not FIRST_MODEL_LAYER <= layer <= LAST_MODEL_LAYER:
            raise ValueError(f"Layer Build 7 non predetto: {layer}")
        if selected.shape != (TOP_K,) or np.any(selected < 0) or np.any(selected >= EXPERTS):
            raise ValueError("Sei expert correnti non validi.")
        ranked_ids = np.argsort(-router_logits, kind="stable")[:RANKED_ROUTER].astype(np.int32)
        ranked_values = router_logits[ranked_ids].astype(np.float32, copy=False)
        kv_statistics = np.asarray([[np.linalg.norm(kv), np.sqrt(np.mean(np.square(kv))),
                                    np.mean(kv), np.std(kv), np.max(np.abs(kv))]], dtype=np.float32)
        scaled_logits = self.router_values_scaler.transform(ranked_values[None, :])[0].astype(np.float32)
        scaled_kv = self.kv_scaler.transform(kv_statistics)[0].astype(np.float32)

        layer_categories = self.layer_encoder.categories_[0]
        matches = np.flatnonzero(layer_categories == layer)
        if len(matches) != 1:
            raise ValueError(f"Layer {layer} assente dall'encoder Build 7")
        layer_offset = EXPERTS + int(matches[0])
        ids_offset = EXPERTS + len(layer_categories)
        numeric_offset = ids_offset + RANKED_ROUTER * EXPERTS
        indices = np.empty((1, 45), dtype=np.int32)
        values = np.empty((1, 45), dtype=np.float32)
        cursor = 0
        history = np.sort(selected)
        indices[0, cursor:cursor + TOP_K] = history
        values[0, cursor:cursor + TOP_K] = 1.0
        cursor += TOP_K
        indices[0, cursor] = layer_offset
        values[0, cursor] = 1.0
        cursor += 1
        indices[0, cursor:cursor + RANKED_ROUTER] = ids_offset + np.arange(RANKED_ROUTER) * EXPERTS + ranked_ids
        values[0, cursor:cursor + RANKED_ROUTER] = 1.0
        cursor += RANKED_ROUTER
        indices[0, cursor:cursor + RANKED_ROUTER] = numeric_offset + np.arange(RANKED_ROUTER)
        values[0, cursor:cursor + RANKED_ROUTER] = scaled_logits
        cursor += RANKED_ROUTER
        indices[0, cursor:cursor + 5] = numeric_offset + RANKED_ROUTER + np.arange(5)
        values[0, cursor:cursor + 5] = scaled_kv
        cursor += 5
        if cursor != 44:
            raise AssertionError("Numero feature Build 7 inatteso")
        indices[0, cursor] = 4411
        values[0, cursor] = 1.0
        return indices, values

    @staticmethod
    def _quantize(hidden: np.ndarray, mean: np.ndarray, std: np.ndarray, scale: float) -> np.ndarray:
        normalized = (hidden - mean) / std
        return np.clip(np.rint(normalized / scale), -127, 127).astype(np.int8)[None, :]

    def predict(self, layer: int, selected: np.ndarray, router_logits: np.ndarray,
                current_hidden: np.ndarray, previous_hidden: np.ndarray, kv: np.ndarray) -> np.ndarray:
        current_index = layer - 3
        previous_index = layer - 4
        state_indices, state_values = self._state_inputs(layer, selected, router_logits, kv)
        current_int8 = self._quantize(
            current_hidden, self.current_means[current_index], self.current_stds[current_index], self.current_scale
        )
        previous_int8 = self._quantize(
            previous_hidden, self.previous_means[previous_index], self.previous_stds[previous_index], self.previous_scale
        )
        # Select the six *columns* P_l(i | selected_j), then average over the
        # selected current experts.  Chained indexing preserves the expected
        # [256, 6] layout; mixed advanced indexing would transpose it.
        transition = self.transition[current_index][:, selected].mean(
            axis=1, dtype=np.float32
        ).astype(np.float16)[None, :]
        scores = self.model({
            "state_feature_indices": state_indices,
            "state_feature_values": state_values,
            "raw_hidden_current": current_int8,
            "raw_hidden_previous": previous_int8,
            "expert_transition_probabilities": transition,
        }, training=False).numpy()[0]
        return np.argsort(-scores, kind="stable")[:TOP_K].astype(np.int32)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Servizio CPU Build 7 Top-4 per TEST_14.")
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-artifacts", type=Path, required=True)
    parser.add_argument("--fixed-nnz-artifacts", type=Path, required=True)
    parser.add_argument("--current-hidden-artifacts", type=Path, required=True)
    parser.add_argument("--previous-hidden-artifacts", type=Path, required=True)
    parser.add_argument("--transition-artifacts", type=Path, required=True)
    parser.add_argument("--current-hidden-scale", type=float, default=0.030650221456692914)
    parser.add_argument("--previous-hidden-scale", type=float, default=0.03066560039370079)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint Build 7 non trovato: {args.checkpoint}")
    args.socket_path.unlink(missing_ok=True)
    print("TEST_14: carico checkpoint e artefatti Build 7 sulla CPU...", flush=True)
    predictor = RuntimePredictor(args)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket_path))
    server.listen(1)
    print(f"TEST_14: predictor pronto su {args.socket_path}", flush=True)
    try:
        while True:
            connection, _ = server.accept()
            with connection:
                while True:
                    try:
                        values = REQUEST.unpack(receive_exact(connection, REQUEST.size))
                    except EOFError:
                        break
                    magic, _pos, _token, layer, request_id = values[:5]
                    if magic != b"T7P2":
                        raise ValueError("Protocollo TEST_14 non riconosciuto")
                    selected = np.asarray(values[5:11], dtype=np.int32)
                    logits_begin = 11
                    current_begin = logits_begin + EXPERTS
                    previous_begin = current_begin + HIDDEN
                    kv_begin = previous_begin + HIDDEN
                    logits = np.asarray(values[logits_begin:current_begin], dtype=np.float32)
                    current = np.asarray(values[current_begin:previous_begin], dtype=np.float32)
                    previous = np.asarray(values[previous_begin:kv_begin], dtype=np.float32)
                    kv = np.asarray(values[kv_begin:kv_begin + KV], dtype=np.float32)
                    started = time.perf_counter()
                    experts = predictor.predict(layer, selected, logits, current, previous, kv)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    connection.sendall(RESPONSE.pack(b"T7R2", request_id, *experts, elapsed_ms))
    finally:
        server.close()
        args.socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
