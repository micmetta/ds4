#!/usr/bin/env python3
"""TEST_10: benchmark isolato delle letture DS4 dal GGUF di prefetch.

Il test non avvia DS4, CUDA né il predictor. Legge invece dal GGUF indicato
con O_DIRECT e usa la stessa forma di una richiesta speculative di DS4:
sei expert, ciascuno formato dai blocchi gate, up e down.
"""

from __future__ import annotations

import json
import mmap
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent

# Dimensioni complessive osservate nelle trace DS4 per il GGUF DeepSeek V4
# Flash usato nei TEST_7/8/9. Le dimensioni singole di gate/up/down e gli
# offset vengono invece letti direttamente dai metadati del GGUF.
EXPERTS_PER_REQUEST = 6
EXPECTED_REQUEST_BYTES = 42_467_328


@dataclass(frozen=True)
class ReadRange:
    """Una lettura gate, up oppure down di un expert."""

    offset: int
    size: int
    tensor_name: str


@dataclass(frozen=True)
class ExpertTensorTable:
    """Offset e dimensione di un tensor GGUF organizzato per expert."""

    name: str
    absolute_offset: int
    bytes_per_expert: int
    expert_count: int


@dataclass(frozen=True)
class ExpertReadPlan:
    """Tabelle reali gate/up/down estratte dall'header del GGUF."""

    layers: dict[int, tuple[ExpertTensorTable, ExpertTensorTable, ExpertTensorTable]]
    gate_bytes_per_expert: int
    up_bytes_per_expert: int
    down_bytes_per_expert: int

    @property
    def bytes_per_expert(self) -> int:
        return (
            self.gate_bytes_per_expert
            + self.up_bytes_per_expert
            + self.down_bytes_per_expert
        )

    @property
    def bytes_per_request(self) -> int:
        return EXPERTS_PER_REQUEST * self.bytes_per_expert

    @property
    def largest_read_bytes(self) -> int:
        return max(
            self.gate_bytes_per_expert,
            self.up_bytes_per_expert,
            self.down_bytes_per_expert,
        )


@dataclass(frozen=True)
class ExpertRequest:
    """Sei expert previsti per un layer, estratti da una trace TEST_9."""

    layer: int
    experts: tuple[int, ...]


def required_environment(name: str) -> str:
    """Restituisce una variabile obbligatoria o interrompe con un messaggio chiaro."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Variabile d'ambiente obbligatoria mancante: {name}")
    return value


def positive_integer_environment(name: str, default: int) -> int:
    """Legge un intero positivo opzionale da una variabile d'ambiente."""
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise SystemExit(f"{name} deve essere un intero positivo.") from error
    if value <= 0:
        raise SystemExit(f"{name} deve essere un intero positivo.")
    return value


def worker_counts_environment() -> list[int]:
    """Legge, valida e ordina i numeri di worker da confrontare."""
    raw_value = os.environ.get("DS4_TEST_10_WORKER_COUNTS", "1,2,3,4")
    try:
        values = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit(
            "DS4_TEST_10_WORKER_COUNTS deve contenere interi separati da virgole, "
            "per esempio 1,2,3,4."
        ) from error
    if not values or any(value < 1 or value > 16 for value in values):
        raise SystemExit("DS4_TEST_10_WORKER_COUNTS accetta valori da 1 a 16.")
    return sorted(set(values))


def next_output() -> Path:
    """Crea una directory di output senza sovrascrivere benchmark precedenti."""
    for suffix in range(1, 1_000_000):
        directory = TEST_DIR / ("output" if suffix == 1 else f"output_{suffix}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            continue
    raise RuntimeError("Non riesco a creare una directory output TEST_10.")


def write_json(path: Path, value: dict) -> None:
    """Salva un report atomico, recuperabile anche dopo Ctrl+C."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def round_down(value: int, alignment: int) -> int:
    """Arrotonda un offset verso il basso al confine O_DIRECT."""
    return value - value % alignment


def round_up(value: int, alignment: int) -> int:
    """Arrotonda un valore verso l'alto all'allineamento GGUF."""
    return ((value + alignment - 1) // alignment) * alignment


def read_u32(stream) -> int:
    """Legge un intero little-endian a 32 bit dall'header GGUF."""
    data = stream.read(4)
    if len(data) != 4:
        raise ValueError("Header GGUF troncato durante una lettura uint32.")
    return int.from_bytes(data, "little")


def read_u64(stream) -> int:
    """Legge un intero little-endian a 64 bit dall'header GGUF."""
    data = stream.read(8)
    if len(data) != 8:
        raise ValueError("Header GGUF troncato durante una lettura uint64.")
    return int.from_bytes(data, "little")


def read_gguf_string(stream) -> str:
    """Legge una stringa GGUF UTF-8 con lunghezza uint64."""
    length = read_u64(stream)
    data = stream.read(length)
    if len(data) != length:
        raise ValueError("Header GGUF troncato durante una stringa.")
    return data.decode("utf-8")


def skip_gguf_value(stream, value_type: int) -> None:
    """Salta un valore metadata GGUF senza interpretarne il contenuto.

    TEST_10 usa soltanto `general.alignment`; tutti gli altri metadati vengono
    saltati per arrivare in modo sicuro alla tabella dei tensor.
    """
    fixed_sizes = {
        0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
        10: 8, 11: 8, 12: 8,
    }
    if value_type in fixed_sizes:
        stream.seek(fixed_sizes[value_type], os.SEEK_CUR)
    elif value_type == 8:  # string
        _ = read_gguf_string(stream)
    elif value_type == 9:  # array
        element_type = read_u32(stream)
        element_count = read_u64(stream)
        for _ in range(element_count):
            skip_gguf_value(stream, element_type)
    else:
        raise ValueError(f"Tipo metadata GGUF non supportato: {value_type}.")


def read_gguf_value(stream, value_type: int) -> int | None:
    """Legge solo i metadati scalari necessari a TEST_10."""
    if value_type == 4:  # uint32: tipo usato da general.alignment.
        return read_u32(stream)
    skip_gguf_value(stream, value_type)
    return None


def parse_expert_read_plan(model: Path) -> ExpertReadPlan:
    """Estrae dal GGUF gli offset reali gate/up/down per gli expert MoE.

    Il formato GGUF conserva gli offset dei tensor rispetto al payload. Dopo
    aver letto header e metadata, TEST_10 individua le tre tabelle di ciascun
    layer: `blk.L.ffn_gate_exps.weight`, `...down...` e `...up...`.
    """
    tensor_infos: list[tuple[str, list[int], int]] = []
    alignment = 32
    with model.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("Il file selezionato non è un GGUF.")
        version = read_u32(stream)
        if version != 3:
            raise ValueError(f"TEST_10 supporta GGUF v3, trovato v{version}.")
        tensor_count = read_u64(stream)
        metadata_count = read_u64(stream)
        for _ in range(metadata_count):
            key = read_gguf_string(stream)
            value_type = read_u32(stream)
            if key == "general.alignment":
                value = read_gguf_value(stream, value_type)
                if value is None or value <= 0:
                    raise ValueError("general.alignment GGUF non valido.")
                alignment = value
            else:
                skip_gguf_value(stream, value_type)
        for _ in range(tensor_count):
            name = read_gguf_string(stream)
            dimensions = [read_u64(stream) for _ in range(read_u32(stream))]
            _tensor_type = read_u32(stream)
            relative_offset = read_u64(stream)
            tensor_infos.append((name, dimensions, relative_offset))
        payload_offset = round_up(stream.tell(), alignment)

    # La fine di ogni tensor coincide con l'offset del tensor successivo nel
    # payload. Così non dobbiamo codificare in TEST_10 le dimensioni dei tipi
    # quantizzati GGUF.
    sorted_infos = sorted(tensor_infos, key=lambda item: item[2])
    tensor_bytes: dict[str, tuple[list[int], int, int]] = {}
    model_size = model.stat().st_size
    for index, (name, dimensions, relative_offset) in enumerate(sorted_infos):
        next_relative_offset = (
            sorted_infos[index + 1][2]
            if index + 1 < len(sorted_infos)
            else model_size - payload_offset
        )
        byte_size = next_relative_offset - relative_offset
        if byte_size <= 0:
            raise ValueError(f"Tensor GGUF con dimensione non positiva: {name}.")
        tensor_bytes[name] = (dimensions, payload_offset + relative_offset, byte_size)

    layer_tables: dict[int, tuple[ExpertTensorTable, ExpertTensorTable, ExpertTensorTable]] = {}
    for layer in range(256):
        names = (
            f"blk.{layer}.ffn_gate_exps.weight",
            f"blk.{layer}.ffn_up_exps.weight",
            f"blk.{layer}.ffn_down_exps.weight",
        )
        if not all(name in tensor_bytes for name in names):
            continue
        tables: list[ExpertTensorTable] = []
        for name in names:
            dimensions, offset, byte_size = tensor_bytes[name]
            if not dimensions:
                raise ValueError(f"Tensor expert senza dimensioni: {name}.")
            expert_count = dimensions[-1]
            if expert_count == 0 or byte_size % expert_count:
                raise ValueError(f"Tensor expert GGUF non divisibile: {name}.")
            tables.append(
                ExpertTensorTable(
                    name=name,
                    absolute_offset=offset,
                    bytes_per_expert=byte_size // expert_count,
                    expert_count=expert_count,
                )
            )
        if len({table.expert_count for table in tables}) != 1:
            raise ValueError(f"Numero expert incoerente nel layer {layer}.")
        layer_tables[layer] = (tables[0], tables[1], tables[2])

    if not layer_tables:
        raise ValueError("Nel GGUF non sono state trovate tabelle expert MoE.")
    first_gate, first_up, first_down = next(iter(layer_tables.values()))
    for gate, up, down in layer_tables.values():
        if (
            gate.bytes_per_expert != first_gate.bytes_per_expert
            or up.bytes_per_expert != first_up.bytes_per_expert
            or down.bytes_per_expert != first_down.bytes_per_expert
        ):
            raise ValueError("Dimensioni expert diverse fra layer MoE non supportate.")
    return ExpertReadPlan(
        layers=layer_tables,
        gate_bytes_per_expert=first_gate.bytes_per_expert,
        up_bytes_per_expert=first_up.bytes_per_expert,
        down_bytes_per_expert=first_down.bytes_per_expert,
    )


def statistics_summary(values: list[float]) -> dict[str, float | int]:
    """Restituisce statistiche semplici, senza dipendenze esterne."""
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "maximum": 0.0}
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "maximum": max(values),
    }


def load_trace_requests(trace: Path, plan: ExpertReadPlan) -> list[ExpertRequest]:
    """Carica richieste reali `learned_prefetch_enqueue` da una trace TEST_9."""
    if not trace.is_file():
        raise SystemExit(f"DS4_TEST_10_TRACE non esiste: {trace}")
    requests: list[ExpertRequest] = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "learned_prefetch_enqueue":
            continue
        layer = record.get("layer")
        experts = record.get("experts")
        if (
            not isinstance(layer, int)
            or layer not in plan.layers
            or not isinstance(experts, list)
            or len(experts) != EXPERTS_PER_REQUEST
            or not all(isinstance(expert, int) for expert in experts)
        ):
            continue
        expert_count = plan.layers[layer][0].expert_count
        if any(expert < 0 or expert >= expert_count for expert in experts):
            continue
        requests.append(ExpertRequest(layer=layer, experts=tuple(experts)))
    if not requests:
        raise SystemExit(
            "DS4_TEST_10_TRACE non contiene richieste learned_prefetch_enqueue valide."
        )
    return requests


def request_ranges(
    plan: ExpertReadPlan,
    request_index: int,
    trace_requests: list[ExpertRequest] | None,
) -> list[ReadRange]:
    """Costruisce 18 letture sugli offset reali di sei expert GGUF.

    Se è disponibile una trace TEST_9, usa gli stessi layer e gli stessi expert
    richiesti dal predictor. Altrimenti sceglie valori deterministici. Offset,
    dimensioni e ordine gate/up/down sono sempre quelli reali del GGUF.
    """
    if trace_requests:
        source_request = trace_requests[request_index % len(trace_requests)]
        layer = source_request.layer
        experts = source_request.experts
    else:
        layers = sorted(plan.layers)
        layer = layers[request_index % len(layers)]
        expert_count = plan.layers[layer][0].expert_count
        experts = tuple(
            (request_index * 37 + expert_position * 53 + 17) % expert_count
            for expert_position in range(EXPERTS_PER_REQUEST)
        )
    gate, up, down = plan.layers[layer]
    ranges: list[ReadRange] = []
    for expert in experts:
        for table, tensor_name in ((gate, "gate"), (up, "up"), (down, "down")):
            ranges.append(
                ReadRange(
                    offset=table.absolute_offset + expert * table.bytes_per_expert,
                    size=table.bytes_per_expert,
                    tensor_name=tensor_name,
                )
            )
    return ranges


def direct_io_flags() -> int:
    """Restituisce i flag necessari; TEST_10 fallisce se O_DIRECT manca."""
    if not hasattr(os, "O_DIRECT"):
        raise SystemExit("TEST_10 richiede O_DIRECT su Linux/WSL.")
    return os.O_RDONLY | os.O_DIRECT


def read_exact(fd: int, buffer: mmap.mmap, offset: int, size: int) -> float:
    """Esegue una preadv O_DIRECT e restituisce il tempo della singola lettura."""
    start = time.perf_counter_ns()
    bytes_read = os.preadv(fd, [memoryview(buffer)[:size]], offset)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    if bytes_read != size:
        raise OSError(f"preadv ha letto {bytes_read} byte invece di {size}.")
    return elapsed_ms


def worker(
    worker_id: int,
    model_path: str,
    plan: ExpertReadPlan,
    trace_requests: list[ExpertRequest] | None,
    groups_per_worker: int,
    warmup_groups_per_worker: int,
    barrier: threading.Barrier,
    results: dict[int, dict],
    errors: list[str],
    error_lock: threading.Lock,
) -> None:
    """Legge gruppi indipendenti: un worker rappresenta un worker SSD DS4."""
    read_times_ms: list[float] = []
    request_times_ms: list[float] = []
    try:
        fd = os.open(model_path, direct_io_flags())
        largest_buffer = mmap.mmap(-1, plan.largest_read_bytes)
        try:
            # Il primo accesso al disco può pagare wake-up, stato energetico o
            # inizializzazione DrvFS. Lo warm-up è volontariamente fuori dai
            # tempi riportati nel benchmark.
            for warmup_group in range(warmup_groups_per_worker):
                request_index = (
                    worker_id * (groups_per_worker + warmup_groups_per_worker)
                    + warmup_group
                )
                for read_range in request_ranges(plan, request_index, trace_requests):
                    _ = read_exact(fd, largest_buffer, read_range.offset, read_range.size)
            barrier.wait()
            for group in range(groups_per_worker):
                # Le richieste di worker diversi sono volutamente distinte.
                request_index = (
                    worker_id * (groups_per_worker + warmup_groups_per_worker)
                    + warmup_groups_per_worker + group
                )
                request_start = time.perf_counter_ns()
                for read_range in request_ranges(plan, request_index, trace_requests):
                    read_times_ms.append(
                        read_exact(fd, largest_buffer, read_range.offset, read_range.size)
                    )
                request_times_ms.append(
                    (time.perf_counter_ns() - request_start) / 1_000_000.0
                )
        finally:
            largest_buffer.close()
            os.close(fd)
        results[worker_id] = {
            "requests_completed": len(request_times_ms),
            "bytes_read": len(request_times_ms) * plan.bytes_per_request,
            "request_ms": statistics_summary(request_times_ms),
            "individual_read_ms": statistics_summary(read_times_ms),
        }
    except BaseException as error:  # Propaga il messaggio nel report principale.
        with error_lock:
            errors.append(f"worker {worker_id}: {error}")
        try:
            barrier.abort()
        except threading.BrokenBarrierError:
            pass


def benchmark_configuration(
    model_path: str,
    plan: ExpertReadPlan,
    trace_requests: list[ExpertRequest] | None,
    worker_count: int,
    groups_per_worker: int,
    warmup_groups_per_worker: int,
) -> dict:
    """Misura una configurazione di concorrenza da 1 a N worker."""
    results: dict[int, dict] = {}
    errors: list[str] = []
    error_lock = threading.Lock()
    barrier = threading.Barrier(worker_count + 1)
    threads = [
        threading.Thread(
            target=worker,
            args=(
                worker_id,
                model_path,
                plan,
                trace_requests,
                groups_per_worker,
                warmup_groups_per_worker,
                barrier,
                results,
                errors,
                error_lock,
            ),
            name=f"test_10_io_worker_{worker_id}",
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

    request_times = [
        item["request_ms"]["mean"]
        for item in results.values()
        if item["request_ms"]["count"]
    ]
    individual_read_times = [
        item["individual_read_ms"]["mean"]
        for item in results.values()
        if item["individual_read_ms"]["count"]
    ]
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
    model_path = required_environment("DS4_TEST_10_MODEL")
    groups_per_worker = positive_integer_environment("DS4_TEST_10_GROUPS_PER_WORKER", 12)
    warmup_groups_per_worker = positive_integer_environment(
        "DS4_TEST_10_WARMUP_GROUPS_PER_WORKER", 2
    )
    worker_counts = worker_counts_environment()
    model = Path(model_path)
    if not model.is_file():
        raise SystemExit(f"DS4_TEST_10_MODEL non punta a un GGUF esistente: {model}")
    file_size = model.stat().st_size
    alignment = max(4096, model.stat().st_blksize)
    plan = parse_expert_read_plan(model)
    trace_value = os.environ.get("DS4_TEST_10_TRACE")
    trace_requests = load_trace_requests(Path(trace_value), plan) if trace_value else None
    if plan.bytes_per_request != EXPECTED_REQUEST_BYTES:
        raise SystemExit(
            "Il GGUF non corrisponde alla forma di richiesta DeepSeek V4 Flash "
            f"attesa: {plan.bytes_per_request} byte invece di {EXPECTED_REQUEST_BYTES}."
        )
    if any(
        size % alignment
        for size in (
            plan.gate_bytes_per_expert,
            plan.up_bytes_per_expert,
            plan.down_bytes_per_expert,
        )
    ):
        raise SystemExit(
            f"Le dimensioni expert non sono allineate a {alignment} byte per O_DIRECT."
        )

    output = next_output()
    report_path = output / "io_benchmark_summary.json"
    report = {
        "experiment": "test_10_isolated_wsl_direct_io_benchmark",
        "status": "running",
        "model": str(model),
        "model_bytes": file_size,
        "direct_io_alignment_bytes": alignment,
        "request_shape": {
            "experts_per_request": EXPERTS_PER_REQUEST,
            "gate_bytes_per_expert": plan.gate_bytes_per_expert,
            "up_bytes_per_expert": plan.up_bytes_per_expert,
            "down_bytes_per_expert": plan.down_bytes_per_expert,
            "bytes_per_expert": plan.bytes_per_expert,
            "bytes_per_request": plan.bytes_per_request,
            "mib_per_request": plan.bytes_per_request / (1024 * 1024),
            "individual_direct_reads_per_request": EXPERTS_PER_REQUEST * 3,
            "moe_layers_found_in_gguf": sorted(plan.layers),
        },
        "configuration": {
            "groups_per_worker": groups_per_worker,
            "warmup_groups_per_worker_not_measured": warmup_groups_per_worker,
            "worker_counts": worker_counts,
            "trace_source": str(Path(trace_value)) if trace_value else None,
            "trace_requests_available": len(trace_requests) if trace_requests else 0,
        },
        "important_limit": (
            "Gli offset e le dimensioni gate/up/down provengono dai metadati del GGUF. "
            "Se DS4_TEST_10_TRACE è impostata, TEST_10 ripete layer ed expert realmente "
            "richiesti dal predictor; altrimenti li sceglie in modo deterministico. In ogni "
            "caso non misura predictor, CUDA o decoding."
        ),
        "results": [],
    }
    write_json(report_path, report)
    print(f"TEST_10 output: {output}")
    print("Benchmark O_DIRECT isolato sul GGUF: "
          f"{plan.bytes_per_request / (1024 * 1024):.2f} MiB per richiesta, "
          f"{groups_per_worker} richieste misurate + {warmup_groups_per_worker} warm-up per worker.")
    try:
        for worker_count in worker_counts:
            print(f"  Misuro {worker_count} worker SSD concorrenti...")
            result = benchmark_configuration(
                str(model),
                plan,
                trace_requests,
                worker_count,
                groups_per_worker,
                warmup_groups_per_worker,
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
        print(f"TEST_10 interrotto: risultati parziali salvati in {report_path}")
        raise
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        write_json(report_path, report)
        raise
    report["status"] = "completed"
    write_json(report_path, report)
    print(f"TEST_10 completato. Report: {report_path}")


if __name__ == "__main__":
    main()
