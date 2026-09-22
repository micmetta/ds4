#!/usr/bin/env python3
"""TEST_10 Windows nativo: stesso piano di letture di TEST_10, senza WSL.

Questo programma deve essere avviato da PowerShell con ``py``. Il codice puo'
risiedere nella cartella del repository WSL, ma il processo Python e le letture
misurate sono Windows nativi: il GGUF viene aperto come ``D:\\...`` e non come
``/mnt/d/...``.
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path


if os.name != "nt":
    raise SystemExit("Questo benchmark deve essere eseguito da Windows con `py`, non da WSL.")


# Riutilizziamo il parser GGUF e il generatore delle 18 letture reali gia'
# verificati nel TEST_10 WSL. L'import non avvia il suo main.
TEST_10_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_10_DIRECTORY))
from test_10_runner import (  # noqa: E402
    EXPECTED_REQUEST_BYTES,
    EXPERTS_PER_REQUEST,
    ExpertReadPlan,
    ExpertRequest,
    load_trace_requests,
    parse_expert_read_plan,
    request_ranges,
    statistics_summary,
)


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_ATTRIBUTE_NORMAL = 0x00000080
MEM_COMMIT = 0x00001000
MEM_RESERVE = 0x00002000
MEM_RELEASE = 0x00008000
PAGE_READWRITE = 0x04
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value


kernel32.CreateFileW.argtypes = (
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
)
kernel32.CreateFileW.restype = wintypes.HANDLE
kernel32.ReadFile.argtypes = (
    wintypes.HANDLE,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.c_void_p,
)
kernel32.ReadFile.restype = wintypes.BOOL
kernel32.GetOverlappedResult.argtypes = (
    wintypes.HANDLE,
    ctypes.c_void_p,
    ctypes.POINTER(wintypes.DWORD),
    wintypes.BOOL,
)
kernel32.GetOverlappedResult.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.VirtualAlloc.argtypes = (
    ctypes.c_void_p,
    ctypes.c_size_t,
    wintypes.DWORD,
    wintypes.DWORD,
)
kernel32.VirtualAlloc.restype = ctypes.c_void_p
kernel32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD)
kernel32.VirtualFree.restype = wintypes.BOOL
kernel32.GetDiskFreeSpaceW.argtypes = (
    wintypes.LPCWSTR,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
)
kernel32.GetDiskFreeSpaceW.restype = wintypes.BOOL


class OVERLAPPED(ctypes.Structure):
    """Parte dell'API Win32 che porta l'offset di una lettura asincrona."""

    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


def win_error(action: str) -> OSError:
    """Crea un errore Windows leggibile per un fallimento dell'API."""
    code = ctypes.get_last_error()
    return OSError(code, f"{action} fallita: {ctypes.FormatError(code)}")


def required_environment(name: str) -> str:
    """Legge una variabile obbligatoria con errore esplicito."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Variabile d'ambiente obbligatoria mancante: {name}")
    return value


def positive_integer_environment(name: str, default: int) -> int:
    """Legge un intero positivo opzionale."""
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise SystemExit(f"{name} deve essere un intero positivo.") from error
    if value <= 0:
        raise SystemExit(f"{name} deve essere un intero positivo.")
    return value


def worker_counts_environment() -> list[int]:
    """Legge la lista 1,2,3,4 dei worker da confrontare."""
    raw_value = os.environ.get("DS4_TEST_10_WINDOWS_WORKER_COUNTS", "1,2,3,4")
    try:
        values = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit(
            "DS4_TEST_10_WINDOWS_WORKER_COUNTS deve contenere interi separati da virgole."
        ) from error
    if not values or any(value < 1 or value > 16 for value in values):
        raise SystemExit("DS4_TEST_10_WINDOWS_WORKER_COUNTS accetta valori da 1 a 16.")
    return sorted(set(values))


def output_directory() -> Path:
    """Crea output, output_2, ... sul filesystem Windows locale."""
    default_root = Path(os.environ["USERPROFILE"]) / "Desktop" / "test_10_windows_native"
    root = Path(os.environ.get("DS4_TEST_10_WINDOWS_OUTPUT_ROOT", str(default_root)))
    root.mkdir(parents=True, exist_ok=True)
    for suffix in range(1, 1_000_000):
        directory = root / ("output" if suffix == 1 else f"output_{suffix}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            continue
    raise RuntimeError("Non riesco a creare una cartella output Windows per TEST_10.")


def write_json(path: Path, value: dict) -> None:
    """Salva il report in modo atomico, cosi' Ctrl+C conserva i risultati parziali."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sector_size(model: Path) -> int:
    """Restituisce il settore logico Windows richiesto da FILE_FLAG_NO_BUFFERING."""
    drive, _ = os.path.splitdrive(str(model))
    root = drive + "\\"
    sectors_per_cluster = wintypes.DWORD()
    bytes_per_sector = wintypes.DWORD()
    free_clusters = wintypes.DWORD()
    total_clusters = wintypes.DWORD()
    if not kernel32.GetDiskFreeSpaceW(
        root,
        ctypes.byref(sectors_per_cluster),
        ctypes.byref(bytes_per_sector),
        ctypes.byref(free_clusters),
        ctypes.byref(total_clusters),
    ):
        raise win_error(f"GetDiskFreeSpaceW({root})")
    return int(bytes_per_sector.value)


class NativeDirectReader:
    """File GGUF Windows aperto con FILE_FLAG_NO_BUFFERING e buffer allineato.

    VirtualAlloc allinea la memoria ad almeno 64 KiB. Gli offset e le
    dimensioni dei tensor del modello sono multipli del settore logico: le
    condizioni necessarie per eseguire il direct I/O senza la cache file di
    Windows.
    """

    def __init__(self, model: Path, buffer_size: int, required_alignment: int) -> None:
        self.alignment = required_alignment
        self.handle = kernel32.CreateFileW(
            str(model),
            GENERIC_READ,
            FILE_SHARE_READ,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            None,
        )
        if self.handle == INVALID_HANDLE_VALUE:
            raise win_error(f"CreateFileW({model})")
        self.buffer = kernel32.VirtualAlloc(
            None, buffer_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE
        )
        if not self.buffer:
            kernel32.CloseHandle(self.handle)
            raise win_error("VirtualAlloc")
        if int(self.buffer) % required_alignment:
            self.close()
            raise RuntimeError("VirtualAlloc non ha restituito un buffer sufficientemente allineato.")

    def read_exact(self, offset: int, size: int) -> float:
        """Legge un blocco GGUF direttamente e ne restituisce il tempo in ms.

        I tensor GGUF non iniziano tutti su un confine da 4 KiB. Windows
        richiede invece offset e dimensione allineati per FILE_FLAG_NO_BUFFERING:
        leggiamo pertanto il piu' piccolo intervallo allineato che contiene il
        blocco gate/up/down richiesto. I pochi byte aggiuntivi non vengono
        usati, ma rendono l'I/O fisico valido e sono parte del lavoro del disco.
        """
        elapsed_ms, _ = self.read_range(offset, size)
        return elapsed_ms

    def read_range(self, offset: int, size: int) -> tuple[float, memoryview]:
        """Legge e restituisce una vista dei soli byte logici richiesti."""
        aligned_offset = offset - offset % self.alignment
        prefix_bytes = offset - aligned_offset
        aligned_size = (
            (prefix_bytes + size + self.alignment - 1) // self.alignment
        ) * self.alignment
        # Ogni ReadFile riceve il proprio offset nel campo OVERLAPPED: non usa
        # e non modifica il cursore condiviso del file. Cio' permette ai worker
        # Windows di leggere realmente in concorrenza.
        overlapped = OVERLAPPED(
            Offset=aligned_offset & 0xFFFFFFFF,
            OffsetHigh=(aligned_offset >> 32) & 0xFFFFFFFF,
        )
        bytes_read = wintypes.DWORD()
        start = time.perf_counter_ns()
        read_started = kernel32.ReadFile(
            self.handle,
            self.buffer,
            aligned_size,
            # Per una chiamata overlapped lpNumberOfBytesRead deve essere
            # NULL: Windows restituisce il dato esclusivamente attraverso
            # GetOverlappedResult quando l'operazione e' conclusa.
            None,
            ctypes.byref(overlapped),
        )
        if not read_started:
            # ERROR_IO_PENDING (997) e' il normale esito di una lettura
            # overlapped non ancora completata. Attendere qui mantiene
            # sequenziali le 18 letture di un singolo worker, come in DS4.
            if ctypes.get_last_error() != 997:
                raise win_error("ReadFile")
        if not kernel32.GetOverlappedResult(
            self.handle, ctypes.byref(overlapped), ctypes.byref(bytes_read), True
        ):
            raise win_error("GetOverlappedResult")
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        if bytes_read.value != aligned_size:
            raise OSError(
                f"ReadFile ha letto {bytes_read.value} byte invece di {aligned_size}."
            )
        payload = (ctypes.c_ubyte * size).from_address(int(self.buffer) + prefix_bytes)
        return elapsed_ms, memoryview(payload).cast("B")

    def close(self) -> None:
        """Rilascia prima il buffer, poi l'handle Windows."""
        if getattr(self, "buffer", None):
            kernel32.VirtualFree(self.buffer, 0, MEM_RELEASE)
            self.buffer = None
        if getattr(self, "handle", INVALID_HANDLE_VALUE) != INVALID_HANDLE_VALUE:
            kernel32.CloseHandle(self.handle)
            self.handle = INVALID_HANDLE_VALUE


def worker(
    worker_id: int,
    model: Path,
    plan: ExpertReadPlan,
    trace_requests: list[ExpertRequest],
    groups_per_worker: int,
    warmup_groups_per_worker: int,
    direct_alignment: int,
    barrier: threading.Barrier,
    results: dict[int, dict],
    errors: list[str],
    error_lock: threading.Lock,
) -> None:
    """Esegue le stesse richieste distinte del worker WSL, da Windows nativo."""
    read_times_ms: list[float] = []
    request_times_ms: list[float] = []
    reader: NativeDirectReader | None = None
    try:
        reader = NativeDirectReader(
            model, plan.largest_read_bytes + direct_alignment, direct_alignment
        )
        for warmup_group in range(warmup_groups_per_worker):
            request_index = worker_id * (groups_per_worker + warmup_groups_per_worker) + warmup_group
            for read_range in request_ranges(plan, request_index, trace_requests):
                reader.read_exact(read_range.offset, read_range.size)
        barrier.wait()
        for group in range(groups_per_worker):
            request_index = (
                worker_id * (groups_per_worker + warmup_groups_per_worker)
                + warmup_groups_per_worker
                + group
            )
            request_start = time.perf_counter_ns()
            for read_range in request_ranges(plan, request_index, trace_requests):
                read_times_ms.append(reader.read_exact(read_range.offset, read_range.size))
            request_times_ms.append((time.perf_counter_ns() - request_start) / 1_000_000.0)
        results[worker_id] = {
            "requests_completed": len(request_times_ms),
            "bytes_read": len(request_times_ms) * plan.bytes_per_request,
            "request_ms": statistics_summary(request_times_ms),
            "individual_read_ms": statistics_summary(read_times_ms),
        }
    except BaseException as error:
        with error_lock:
            errors.append(f"worker {worker_id}: {error}")
        try:
            barrier.abort()
        except threading.BrokenBarrierError:
            pass
    finally:
        if reader is not None:
            reader.close()


def benchmark_configuration(
    model: Path,
    plan: ExpertReadPlan,
    trace_requests: list[ExpertRequest],
    worker_count: int,
    groups_per_worker: int,
    warmup_groups_per_worker: int,
    direct_alignment: int,
) -> dict:
    """Misura una configurazione con N worker Windows concorrenti."""
    results: dict[int, dict] = {}
    errors: list[str] = []
    error_lock = threading.Lock()
    barrier = threading.Barrier(worker_count + 1)
    threads = [
        threading.Thread(
            target=worker,
            args=(
                worker_id, model, plan, trace_requests, groups_per_worker,
                warmup_groups_per_worker, direct_alignment, barrier, results,
                errors, error_lock,
            ),
            name=f"test_10_windows_io_worker_{worker_id}",
        )
        for worker_id in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    start = time.perf_counter_ns()
    for thread in threads:
        thread.join()
    elapsed_seconds = (time.perf_counter_ns() - start) / 1_000_000_000.0
    if errors:
        raise RuntimeError("; ".join(errors))

    request_times = [item["request_ms"]["mean"] for item in results.values()]
    individual_read_times = [item["individual_read_ms"]["mean"] for item in results.values()]
    total_bytes = sum(int(item["bytes_read"]) for item in results.values())
    return {
        "worker_count": worker_count,
        "groups_per_worker": groups_per_worker,
        "warmup_groups_per_worker_not_measured": warmup_groups_per_worker,
        "requests_completed": worker_count * groups_per_worker,
        "individual_reads_completed": worker_count * groups_per_worker * EXPERTS_PER_REQUEST * 3,
        "total_bytes_read": total_bytes,
        "total_mib_read": total_bytes / (1024 * 1024),
        "elapsed_seconds": elapsed_seconds,
        "aggregate_mib_per_second": total_bytes / (1024 * 1024) / elapsed_seconds,
        "mean_request_ms_by_worker": statistics_summary(request_times),
        "mean_individual_read_ms_by_worker": statistics_summary(individual_read_times),
        "per_worker": {str(worker_id): results[worker_id] for worker_id in sorted(results)},
    }


def main() -> None:
    model = Path(required_environment("DS4_TEST_10_WINDOWS_MODEL"))
    trace = Path(required_environment("DS4_TEST_10_WINDOWS_TRACE"))
    groups_per_worker = positive_integer_environment("DS4_TEST_10_WINDOWS_GROUPS_PER_WORKER", 12)
    warmup_groups_per_worker = positive_integer_environment(
        "DS4_TEST_10_WINDOWS_WARMUP_GROUPS_PER_WORKER", 2
    )
    worker_counts = worker_counts_environment()
    if not model.is_file():
        raise SystemExit(f"DS4_TEST_10_WINDOWS_MODEL non punta a un GGUF: {model}")
    if not trace.is_file():
        raise SystemExit(f"DS4_TEST_10_WINDOWS_TRACE non esiste: {trace}")

    plan = parse_expert_read_plan(model)
    if plan.bytes_per_request != EXPECTED_REQUEST_BYTES:
        raise SystemExit(
            f"GGUF inatteso: richiesta da {plan.bytes_per_request} byte, attesi {EXPECTED_REQUEST_BYTES}."
        )
    direct_alignment = sector_size(model)
    dimensions = (
        plan.gate_bytes_per_expert,
        plan.up_bytes_per_expert,
        plan.down_bytes_per_expert,
    )
    if any(size % direct_alignment for size in dimensions):
        raise SystemExit(
            f"I blocchi expert non sono multipli del settore Windows da {direct_alignment} byte."
        )
    trace_requests = load_trace_requests(trace, plan)
    output = output_directory()
    report_path = output / "io_benchmark_windows_summary.json"
    report = {
        "experiment": "test_10_native_windows_direct_io_benchmark",
        "status": "running",
        "execution": "Python e ReadFile nativi Windows; GGUF aperto direttamente dal percorso Windows.",
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "direct_io": {"api": "CreateFileW + FILE_FLAG_NO_BUFFERING + ReadFile", "sector_alignment_bytes": direct_alignment},
        "request_shape": {
            "experts_per_request": EXPERTS_PER_REQUEST,
            "gate_bytes_per_expert": plan.gate_bytes_per_expert,
            "up_bytes_per_expert": plan.up_bytes_per_expert,
            "down_bytes_per_expert": plan.down_bytes_per_expert,
            "bytes_per_expert": plan.bytes_per_expert,
            "bytes_per_request": plan.bytes_per_request,
            "mib_per_request": plan.bytes_per_request / (1024 * 1024),
            "individual_direct_reads_per_request": EXPERTS_PER_REQUEST * 3,
        },
        "configuration": {
            "groups_per_worker": groups_per_worker,
            "warmup_groups_per_worker_not_measured": warmup_groups_per_worker,
            "worker_counts": worker_counts,
            "trace_source_read_before_timing": str(trace),
            "trace_requests_available": len(trace_requests),
        },
        "results": [],
    }
    write_json(report_path, report)
    print(f"TEST_10 Windows output: {output}")
    print(
        "Benchmark Windows FILE_FLAG_NO_BUFFERING: "
        f"{plan.bytes_per_request / (1024 * 1024):.2f} MiB per richiesta, "
        f"{groups_per_worker} richieste misurate + {warmup_groups_per_worker} warm-up per worker."
    )
    try:
        for worker_count in worker_counts:
            print(f"  Misuro {worker_count} worker SSD Windows concorrenti...")
            result = benchmark_configuration(
                model, plan, trace_requests, worker_count, groups_per_worker,
                warmup_groups_per_worker, direct_alignment,
            )
            report["results"].append(result)
            write_json(report_path, report)
            print(
                f"    {result['aggregate_mib_per_second']:.2f} MiB/s aggregati, "
                f"{result['elapsed_seconds']:.3f} s."
            )
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        write_json(report_path, report)
        print(f"TEST_10 Windows interrotto: risultati parziali in {report_path}")
        raise
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        write_json(report_path, report)
        raise
    report["status"] = "completed"
    write_json(report_path, report)
    print(f"TEST_10 Windows completato. Report: {report_path}")


if __name__ == "__main__":
    main()
