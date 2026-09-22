#!/usr/bin/env python3
"""TEST_13 Windows service: riceve layer + sei expert e invia i pesi reali."""
from __future__ import annotations
import os
import socket
import struct
import sys
import threading
from pathlib import Path
if os.name != "nt": raise SystemExit("Avviare da PowerShell.")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test_10" / "windows_native"))
from test_10_windows_native import NativeDirectReader, parse_expert_read_plan, sector_size
REQ = struct.Struct("<II6i")
MAGIC = 0x31335444

def recv_full(connection: socket.socket, size: int) -> bytes | None:
    """Riceve esattamente ``size`` byte o segnala una connessione chiusa."""
    data = bytearray(size)
    view = memoryview(data)
    received = 0
    while received < size:
        count = connection.recv_into(view[received:])
        if not count:
            return None
        received += count
    return bytes(data)

def handle(connection: socket.socket, model: Path, plan, alignment: int) -> None:
    """Serve una richiesta DS4: sei expert, tre tensori per expert."""
    reader = NativeDirectReader(model, plan.largest_read_bytes + alignment, alignment)
    try:
        data = recv_full(connection, REQ.size)
        if data is None:
            return
        magic, layer, *experts = REQ.unpack(data)
        if magic != MAGIC or layer not in plan.layers:
            return
        gate, up, down = plan.layers[layer]
        for expert in experts:
            if expert < 0 or expert >= gate.expert_count:
                return
            for table in (gate, up, down):
                offset = table.absolute_offset + expert * table.bytes_per_expert
                _, block = reader.read_range(offset, table.bytes_per_expert)
                connection.sendall(block)
    finally:
        reader.close()
        connection.close()

def main() -> None:
    model = Path(os.environ["DS4_TEST_13_WINDOWS_MODEL"])
    if not model.is_file():
        raise SystemExit(f"GGUF Windows non trovato: {model}")
    port = int(os.environ.get("DS4_TEST_13_WINDOWS_PORT", "38113"))
    plan = parse_expert_read_plan(model)
    alignment = sector_size(model)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen()
        print(f"TEST_13 servizio Windows pronto su 0.0.0.0:{port}.", flush=True)
        while True:
            connection, _ = server.accept()
            threading.Thread(
                target=handle, args=(connection, model, plan, alignment), daemon=True
            ).start()

if __name__ == "__main__": main()
