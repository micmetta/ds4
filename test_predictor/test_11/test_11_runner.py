#!/usr/bin/env python3
"""Client WSL per misurare Windows GGUF -> TCP localhost -> WSL."""
from __future__ import annotations

import json
import os
import socket
import statistics
import struct
import threading
import time
from pathlib import Path

REQUEST_BYTES = 42_467_328
REQUEST = struct.Struct("<I")
METRICS = struct.Struct("<dd")
STOP = 0xFFFFFFFF
TEST_DIR = Path(__file__).resolve().parent


def integers(name: str, default: str) -> list[int]:
    try:
        result = sorted({int(value) for value in os.environ.get(name, default).split(",")})
    except ValueError as error:
        raise SystemExit(f"{name} deve contenere interi separati da virgole.") from error
    if not result or any(value < 1 for value in result):
        raise SystemExit(f"{name} deve contenere interi positivi.")
    return result


def default_windows_host() -> str:
    """Trova il gateway Windows della VM WSL2, se localhost non e' inoltrato."""
    try:
        for line in Path("/proc/net/route").read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 3 and fields[1] == "00000000":
                return socket.inet_ntoa(struct.pack("<I", int(fields[2], 16)))
    except OSError:
        pass
    return "127.0.0.1"


def output_directory() -> Path:
    for suffix in range(1, 1_000_000):
        path = TEST_DIR / ("output" if suffix == 1 else f"output_{suffix}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            pass
    raise RuntimeError("Impossibile creare output TEST_11.")


def recv_exact(connection: socket.socket, size: int) -> None:
    """Riceve e scarta i payload, ma misura il vero attraversamento in WSL."""
    buffer = bytearray(min(size, 1024 * 1024))
    view = memoryview(buffer)
    remaining = size
    while remaining:
        received = connection.recv_into(view[: min(remaining, len(view))])
        if not received:
            raise ConnectionError("Il servizio Windows ha chiuso la connessione.")
        remaining -= received


def one_worker(worker_id: int, groups: int, warmup: int, host: str, port: int, barrier, results, errors) -> None:
    try:
        with socket.create_connection((host, port), timeout=10) as connection:
            connection.settimeout(120)
            for request_index in range(worker_id * (groups + warmup), worker_id * (groups + warmup) + warmup):
                connection.sendall(REQUEST.pack(request_index))
                recv_exact(connection, REQUEST_BYTES)
                recv_exact(connection, METRICS.size)
            barrier.wait()
            end_to_end, read_ms, send_ms = [], [], []
            for group in range(groups):
                index = worker_id * (groups + warmup) + warmup + group
                start = time.perf_counter_ns()
                connection.sendall(REQUEST.pack(index))
                recv_exact(connection, REQUEST_BYTES)
                header = bytearray(METRICS.size)
                view = memoryview(header)
                received = 0
                while received < METRICS.size:
                    received += connection.recv_into(view[received:])
                disk, sent = METRICS.unpack(header)
                end_to_end.append((time.perf_counter_ns() - start) / 1_000_000.0)
                read_ms.append(disk)
                send_ms.append(sent)
            results[worker_id] = {"end_to_end_ms": end_to_end, "windows_read_ms": read_ms, "windows_send_ms": send_ms}
    except BaseException as error:
        errors.append(f"worker {worker_id}: {error}")
        barrier.abort()


def summary(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "p95": sorted(values)[max(0, (95*len(values)+99)//100-1)]} if values else {"mean": 0, "p95": 0}


def main() -> None:
    host = os.environ.get("DS4_TEST_11_HOST", default_windows_host())
    port = int(os.environ.get("DS4_TEST_11_PORT", "38111"))
    groups = int(os.environ.get("DS4_TEST_11_GROUPS_PER_WORKER", "12"))
    warmup = int(os.environ.get("DS4_TEST_11_WARMUP_GROUPS_PER_WORKER", "2"))
    workers = integers("DS4_TEST_11_WORKER_COUNTS", "1,2,3,4,5,6,7")
    output = output_directory()
    report = {"experiment": "test_11_windows_to_wsl_localhost", "status": "running", "request_bytes": REQUEST_BYTES, "results": []}
    report_path = output / "windows_to_wsl_summary.json"
    print(f"TEST_11 output: {output}; host Windows: {host}:{port}")
    try:
        for count in workers:
            print(f"  Misuro {count} flussi Windows -> WSL concorrenti...")
            results, errors = {}, []
            barrier = threading.Barrier(count + 1)
            threads = [threading.Thread(target=one_worker, args=(i, groups, warmup, host, port, barrier, results, errors)) for i in range(count)]
            for thread in threads: thread.start()
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                for thread in threads: thread.join()
                raise RuntimeError("; ".join(errors) or "Connessione al servizio Windows non riuscita.")
            start = time.perf_counter_ns()
            for thread in threads: thread.join()
            elapsed = (time.perf_counter_ns()-start)/1e9
            if errors: raise RuntimeError("; ".join(errors))
            all_e2e = [item for result in results.values() for item in result["end_to_end_ms"]]
            all_read = [item for result in results.values() for item in result["windows_read_ms"]]
            all_send = [item for result in results.values() for item in result["windows_send_ms"]]
            total = count * groups * REQUEST_BYTES
            item = {"worker_count": count, "requests": count*groups, "elapsed_seconds": elapsed, "aggregate_mib_per_second": total/1048576/elapsed, "end_to_end_ms": summary(all_e2e), "windows_read_ms": summary(all_read), "windows_send_ms": summary(all_send)}
            report["results"].append(item)
            report_path.write_text(json.dumps(report, indent=2)+"\n")
            print(f"    {item['aggregate_mib_per_second']:.2f} MiB/s end-to-end; lettura Windows {item['windows_read_ms']['mean']:.1f} ms; invio {item['windows_send_ms']['mean']:.1f} ms.")
    finally:
        report["status"] = "completed" if not errors else "failed"
        report_path.write_text(json.dumps(report, indent=2)+"\n")
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.sendall(REQUEST.pack(STOP))
    print(f"TEST_11 completato. Report: {report_path}")


if __name__ == "__main__":
    main()
