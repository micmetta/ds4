#!/usr/bin/env python3
"""TEST_12: ricezione Windows -> WSL direttamente in memoria CUDA pinned."""
from __future__ import annotations

import ctypes
import json
import os
import socket
import statistics
import struct
import threading
import time
from pathlib import Path

BYTES = 42_467_328
REQUEST, METRICS, STOP = struct.Struct("<I"), struct.Struct("<dd"), 0xFFFFFFFF
TEST_DIR = Path(__file__).resolve().parent


def windows_host() -> str:
    for line in Path("/proc/net/route").read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 3 and fields[1] == "00000000":
            return socket.inet_ntoa(struct.pack("<I", int(fields[2], 16)))
    return "127.0.0.1"


def output() -> Path:
    for number in range(1, 1_000_000):
        path = TEST_DIR / ("output" if number == 1 else f"output_{number}")
        try: path.mkdir(); return path
        except FileExistsError: pass
    raise RuntimeError("Impossibile creare output TEST_12.")


class PinnedBuffer:
    """Buffer cudaMallocHost: stesso tipo di memoria pinned usato dai worker DS4."""
    def __init__(self, size: int) -> None:
        self.runtime = ctypes.CDLL("libcudart.so")
        self.runtime.cudaMallocHost.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
        self.runtime.cudaFreeHost.argtypes = (ctypes.c_void_p,)
        self.pointer = ctypes.c_void_p()
        error = self.runtime.cudaMallocHost(ctypes.byref(self.pointer), size)
        if error: raise RuntimeError(f"cudaMallocHost fallita, codice CUDA {error}.")
        self.view = memoryview((ctypes.c_ubyte * size).from_address(self.pointer.value)).cast("B")
    def close(self) -> None:
        if self.pointer.value:
            self.runtime.cudaFreeHost(self.pointer); self.pointer = ctypes.c_void_p()


def receive(connection: socket.socket, view: memoryview, size: int) -> None:
    position = 0
    while position < size:
        received = connection.recv_into(view[position:size])
        if not received: raise ConnectionError("Servizio Windows chiuso.")
        position += received


def worker(identifier, groups, warmup, host, port, barrier, results, errors) -> None:
    pinned = None
    try:
        pinned = PinnedBuffer(BYTES)
        with socket.create_connection((host, port), timeout=10) as connection:
            connection.settimeout(120)
            header = bytearray(METRICS.size)
            for index in range(identifier*(groups+warmup), identifier*(groups+warmup)+warmup):
                connection.sendall(REQUEST.pack(index)); receive(connection, pinned.view, BYTES); receive(connection, memoryview(header), METRICS.size)
            barrier.wait()
            totals=[]; reads=[]; sends=[]
            for group in range(groups):
                index=identifier*(groups+warmup)+warmup+group; start=time.perf_counter_ns()
                connection.sendall(REQUEST.pack(index)); receive(connection,pinned.view,BYTES); receive(connection,memoryview(header),METRICS.size)
                read, sent=METRICS.unpack(header); totals.append((time.perf_counter_ns()-start)/1e6); reads.append(read); sends.append(sent)
            results[identifier]={"end_to_end_ms":totals,"windows_read_ms":reads,"windows_send_ms":sends}
    except BaseException as error:
        errors.append(f"worker {identifier}: {error}")
        try: barrier.abort()
        except threading.BrokenBarrierError: pass
    finally:
        if pinned: pinned.close()


def stats(values): return {"mean":statistics.mean(values),"p95":sorted(values)[max(0,(95*len(values)+99)//100-1)]}


def main() -> None:
    host=os.environ.get("DS4_TEST_12_HOST",windows_host()); port=int(os.environ.get("DS4_TEST_12_PORT","38111"))
    groups=int(os.environ.get("DS4_TEST_12_GROUPS_PER_WORKER","12")); warmup=int(os.environ.get("DS4_TEST_12_WARMUP_GROUPS_PER_WORKER","2"))
    counts=sorted({int(x) for x in os.environ.get("DS4_TEST_12_WORKER_COUNTS","1,2,3,4,5,6,7").split(",")})
    directory=output(); path=directory/"windows_to_wsl_pinned_summary.json"; report={"experiment":"test_12_windows_to_wsl_cuda_pinned","status":"running","request_bytes":BYTES,"host":host,"results":[]}
    print(f"TEST_12 output: {directory}; host Windows: {host}:{port}")
    try:
        for count in counts:
            print(f"  Misuro {count} flussi ricevuti in CUDA pinned RAM...")
            results={}; errors=[]; barrier=threading.Barrier(count+1)
            threads=[threading.Thread(target=worker,args=(i,groups,warmup,host,port,barrier,results,errors)) for i in range(count)]
            [thread.start() for thread in threads]
            try: barrier.wait()
            except threading.BrokenBarrierError:
                [thread.join() for thread in threads]; raise RuntimeError("; ".join(errors))
            started=time.perf_counter_ns(); [thread.join() for thread in threads]
            if errors: raise RuntimeError("; ".join(errors))
            elapsed=(time.perf_counter_ns()-started)/1e9; all_total=[x for r in results.values() for x in r["end_to_end_ms"]]; all_read=[x for r in results.values() for x in r["windows_read_ms"]]; all_send=[x for r in results.values() for x in r["windows_send_ms"]]
            item={"worker_count":count,"requests":count*groups,"elapsed_seconds":elapsed,"aggregate_mib_per_second":count*groups*BYTES/1048576/elapsed,"end_to_end_ms":stats(all_total),"windows_read_ms":stats(all_read),"windows_send_ms":stats(all_send)}
            report["results"].append(item); path.write_text(json.dumps(report,indent=2)+"\n"); print(f"    {item['aggregate_mib_per_second']:.2f} MiB/s end-to-end; ricezione direttamente in CUDA pinned RAM.")
    finally:
        report["status"]="completed"; path.write_text(json.dumps(report,indent=2)+"\n")
    with socket.create_connection((host,port),timeout=10) as connection: connection.sendall(REQUEST.pack(STOP))
    print(f"TEST_12 completato. Report: {path}")

if __name__ == "__main__": main()
