#!/usr/bin/env python3
"""Servizio Windows: legge dal GGUF e invia i blocchi a un client WSL."""
from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

if os.name != "nt":
    raise SystemExit("Avviare questo servizio da PowerShell con `py`.")

TEST10_WINDOWS = Path(__file__).resolve().parents[2] / "test_10" / "windows_native"
sys.path.insert(0, str(TEST10_WINDOWS))
from test_10_windows_native import (  # noqa: E402
    EXPECTED_REQUEST_BYTES, NativeDirectReader, load_trace_requests,
    parse_expert_read_plan, request_ranges, sector_size,
)

REQUEST = struct.Struct("<I")
METRICS = struct.Struct("<dd")  # millisecondi lettura, millisecondi invio
STOP = 0xFFFFFFFF


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Variabile obbligatoria mancante: {name}")
    return value


def handle(client: socket.socket, model: Path, plan, trace, alignment: int, stop: threading.Event) -> None:
    """Una connessione e' un worker: riceve indice, legge e invia 40,5 MiB."""
    reader = NativeDirectReader(model, plan.largest_read_bytes + alignment, alignment)
    try:
        while not stop.is_set():
            message = client.recv(REQUEST.size)
            if len(message) != REQUEST.size:
                return
            (request_index,) = REQUEST.unpack(message)
            if request_index == STOP:
                stop.set()
                return
            read_ms = 0.0
            send_start = time.perf_counter_ns()
            # Manteniamo l'ordine gate/up/down e inviamo esattamente i byte
            # logici, non il piccolo prefisso aggiunto per l'allineamento.
            for read_range in request_ranges(plan, request_index, trace):
                elapsed_ms, data = reader.read_range(read_range.offset, read_range.size)
                read_ms += elapsed_ms
                client.sendall(data)
            send_ms = (time.perf_counter_ns() - send_start) / 1_000_000.0
            client.sendall(METRICS.pack(read_ms, send_ms))
    finally:
        reader.close()
        client.close()


def main() -> None:
    model = Path(required("DS4_TEST_11_WINDOWS_MODEL"))
    trace_path = Path(required("DS4_TEST_11_WINDOWS_TRACE"))
    port = int(os.environ.get("DS4_TEST_11_PORT", "38111"))
    if not model.is_file() or not trace_path.is_file():
        raise SystemExit("Modello GGUF o trace non esistente.")
    plan = parse_expert_read_plan(model)
    if plan.bytes_per_request != EXPECTED_REQUEST_BYTES:
        raise SystemExit("GGUF con forma expert inattesa.")
    trace = load_trace_requests(trace_path, plan)
    alignment = sector_size(model)
    stop = threading.Event()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen()
        server.settimeout(0.5)
        print(f"TEST_11 servizio Windows pronto su 0.0.0.0:{port}.", flush=True)
        threads: list[threading.Thread] = []
        try:
            while not stop.is_set():
                try:
                    client, _ = server.accept()
                except TimeoutError:
                    continue
                thread = threading.Thread(
                    target=handle, args=(client, model, plan, trace, alignment, stop), daemon=True
                )
                thread.start()
                threads.append(thread)
        finally:
            for thread in threads:
                thread.join(timeout=2)
    print("TEST_11 servizio Windows terminato.", flush=True)


if __name__ == "__main__":
    main()
