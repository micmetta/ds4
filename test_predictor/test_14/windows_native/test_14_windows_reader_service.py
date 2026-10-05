#!/usr/bin/env python3
"""TEST_14 Windows service: same native GGUF reader used by TEST_13."""
from __future__ import annotations

import os
import socket
import struct
import sys
import threading
from pathlib import Path


if os.name != "nt":
    raise SystemExit("Avviare da PowerShell.")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test_10" / "windows_native"))
from test_10_windows_native import NativeDirectReader, parse_expert_read_plan, sector_size


# TEST_14 adds the number of predicted experts to the header.  The payload
# remains intentionally small.  TEST_14 uses Top-6..10 groups; TEST_15's
# shared cache requests one expert at a time through the same bridge.
REQUEST_HEADER = struct.Struct("<III")
MIN_PREFETCH_TOP_K = 1
MAX_PREFETCH_TOP_K = 10
MAGIC = 0x31335444


def receive_exact(connection: socket.socket, size: int) -> bytes | None:
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
    reader = NativeDirectReader(model, plan.largest_read_bytes + alignment, alignment)
    try:
        payload = receive_exact(connection, REQUEST_HEADER.size)
        if payload is None:
            return
        magic, layer, expert_count = REQUEST_HEADER.unpack(payload)
        if magic != MAGIC or layer not in plan.layers or not MIN_PREFETCH_TOP_K <= expert_count <= MAX_PREFETCH_TOP_K:
            return
        experts_payload = receive_exact(connection, expert_count * struct.calcsize("<i"))
        if experts_payload is None:
            return
        experts = struct.unpack(f"<{expert_count}i", experts_payload)
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
    model = Path(os.environ["DS4_TEST_14_WINDOWS_MODEL"])
    if not model.is_file():
        raise SystemExit(f"GGUF Windows non trovato: {model}")
    port = int(os.environ.get("DS4_TEST_14_WINDOWS_PORT", "38114"))
    plan = parse_expert_read_plan(model)
    alignment = sector_size(model)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen()
        print(f"TEST_14 servizio Windows pronto su 0.0.0.0:{port}.", flush=True)
        while True:
            connection, _ = server.accept()
            threading.Thread(
                target=handle, args=(connection, model, plan, alignment), daemon=True
            ).start()


if __name__ == "__main__":
    main()
