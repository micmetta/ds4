#!/usr/bin/env python3
"""TEST_9: benchmark A/B/C ripetuto con audit automatico della banca pinned.

TEST_9 riusa il predictor CPU e il learned prefetch verificati in TEST_7, ma
esegue autonomamente A, B e C per piu' ripetizioni come TEST_8. Dopo ogni fase
C legge subito la sua trace e salva il report della politica pinned-RAM nella
stessa directory della ripetizione. L'opzione --audit-trace conserva anche la
modalita' read-only per analizzare una trace storica senza lanciare DS4.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
TEST7_DIR = ROOT / "test_predictor" / "test_7"
sys.path.insert(0, str(TEST7_DIR))

# TEST_7 resta la sola implementazione del protocollo socket, del servizio
# TensorFlow sulla CPU e dell'invocazione di DS4. TEST_9 aggiunge il benchmark
# ripetuto e l'audit automatico degli slot pinned.
from test_7_runner import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    correct_predictions_prefetch_availability_summary,
    learned_trace_summary,
    prefetch_trace_summary,
    run_ds4,
    start_service,
    stop_service,
)


def required_environment(name: str) -> str:
    """Restituisce una variabile obbligatoria o interrompe con un messaggio chiaro."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Variabile d'ambiente obbligatoria mancante: {name}")
    return value


def next_output() -> Path:
    """Crea un output TEST_9 nuovo senza sovrascrivere run precedenti."""
    for suffix in range(1, 1_000_000):
        directory = TEST_DIR / ("output" if suffix == 1 else f"output_{suffix}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            continue
    raise RuntimeError("Non riesco a creare una directory output TEST_9.")


def write_json(path: Path, value: dict) -> None:
    """Salva in modo atomico per conservare il progresso anche dopo Ctrl+C."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    """Legge tutti i record JSON non vuoti della trace del worker pinned-RAM."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(records: list[dict]) -> dict:
    """Ricostruisce disponibilità degli slot e starvation del worker.

    Le trace create con ``DS4_TEST_9_PINNED_DIAGNOSTICS=1`` includono uno
    snapshot atomico della banca pinned per ogni evento del worker. Le trace
    storiche restano valide: i campi diagnostici vengono semplicemente
    riportati come non disponibili.
    """
    event_counts = Counter(record.get("event", "unknown") for record in records)
    bank_full_by_layer = Counter()
    stale_by_layer = Counter()
    stale_queued_discarded_by_layer = Counter()
    stale_queued_discarded_by_worker_by_layer = Counter()
    reading_marked_too_late_by_layer = Counter()
    too_late_freed_by_layer = Counter()
    bank_full_then_stale_by_layer = Counter()
    bank_full_during_partial_reuse_by_layer = Counter()
    generated_by_layer = Counter()
    queued_by_layer = Counter()
    started_by_layer = Counter()
    ready_by_layer = Counter()
    failed_by_layer = Counter()
    bank_full_reason_by_layer: dict[int, Counter] = defaultdict(Counter)
    bank_full_reason_total = Counter()
    bank_full_slot_states = Counter()
    bank_full_lower_loading_slots: list[float] = []
    bank_full_slot_ages: list[float] = []
    worker_wait_ms: list[float] = []
    worker_read_ms: list[float] = []
    worker_total_ms: list[float] = []
    worker_started_by_id = Counter()
    worker_ready_by_id = Counter()
    worker_read_ms_by_id: dict[int, list[float]] = defaultdict(list)
    loading_slots_observed: list[float] = []
    lower_layer_loading_observed: list[float] = []
    pinned_slots_observed: list[float] = []
    pinned_bytes_observed: list[float] = []
    pending_bank_full: dict[int, list[dict]] = defaultdict(list)

    def numeric(value: object) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None

    def summary(values: list[float]) -> dict[str, float | int]:
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

    def bank_full_reason(snapshot: object) -> tuple[str, list[str], list[float]]:
        if not isinstance(snapshot, dict):
            return "snapshot_non_disponibile", [], []
        slots = snapshot.get("same_layer_slots")
        if not isinstance(slots, list) or not slots:
            return "snapshot_non_disponibile", [], []
        states = [str(slot.get("state", "UNKNOWN")) for slot in slots if isinstance(slot, dict)]
        ages = [value for slot in slots if isinstance(slot, dict)
                if (value := numeric(slot.get("age_ms"))) is not None]
        if states and all(state == "QUEUED" for state in states):
            return "tutti_slot_queued", states, ages
        if states and all(state == "READING" for state in states):
            return "tutti_slot_reading", states, ages
        if states and all(state == "LOADING" for state in states):
            return "tutti_slot_loading_formato_storico", states, ages
        if "READING" in states:
            return "almeno_uno_slot_reading", states, ages
        if "QUEUED" in states:
            return "almeno_uno_slot_queued", states, ages
        if "TOO_LATE" in states:
            return "almeno_uno_slot_too_late", states, ages
        if "CONSUMING" in states:
            return "almeno_uno_slot_consuming", states, ages
        if "READY" in states:
            return "almeno_uno_slot_ready", states, ages
        if "FAILED" in states:
            return "almeno_uno_slot_failed", states, ages
        return "slot_non_empty_misto", states, ages

    for position, record in enumerate(records):
        event = record.get("event")
        layer = record.get("layer")
        if not isinstance(layer, int):
            continue
        if event == "learned_prefetch_enqueue":
            generated_by_layer[layer] += 1
        elif event == "prefetch_worker_queued":
            queued_by_layer[layer] += 1
        elif event == "prefetch_worker_started":
            started_by_layer[layer] += 1
            if isinstance(record.get("worker_id"), int):
                worker_started_by_id[record["worker_id"]] += 1
            if (value := numeric(record.get("queue_to_worker_ms"))) is not None:
                worker_wait_ms.append(value)
        elif event == "prefetch_worker_ready":
            ready_by_layer[layer] += 1
            if isinstance(record.get("worker_id"), int):
                worker_ready_by_id[record["worker_id"]] += 1
            if (value := numeric(record.get("worker_read_ms"))) is not None:
                worker_read_ms.append(value)
                if isinstance(record.get("worker_id"), int):
                    worker_read_ms_by_id[record["worker_id"]].append(value)
            if (value := numeric(record.get("queue_to_ready_ms"))) is not None:
                worker_total_ms.append(value)
        elif event == "prefetch_worker_failed":
            failed_by_layer[layer] += 1

        snapshot = record.get("pinned_bank_snapshot")
        if isinstance(snapshot, dict):
            if (value := numeric(snapshot.get("bank_slots", {}).get("loading"))) is not None:
                loading_slots_observed.append(value)
            if (value := numeric(snapshot.get("lower_layer_loading_slots"))) is not None:
                lower_layer_loading_observed.append(value)
            if (value := numeric(snapshot.get("pinned_slots_allocated"))) is not None:
                pinned_slots_observed.append(value)
            if (value := numeric(snapshot.get("pinned_bytes_estimated"))) is not None:
                pinned_bytes_observed.append(value)

        if event == "prefetch_bank_full":
            bank_full_by_layer[layer] += 1
            reason, states, ages = bank_full_reason(snapshot)
            bank_full_reason_total[reason] += 1
            bank_full_reason_by_layer[layer][reason] += 1
            bank_full_slot_states[" + ".join(states) if states else "non disponibili"] += 1
            bank_full_slot_ages.extend(ages)
            if isinstance(snapshot, dict):
                if (value := numeric(snapshot.get("lower_layer_loading_slots"))) is not None:
                    bank_full_lower_loading_slots.append(value)
            pending_bank_full[layer].append({
                "trace_position": position,
                "target_token_index": record.get("token_index"),
            })
        elif event == "prefetch_stale_discarded":
            stale_by_layer[layer] += 1
        elif event == "prefetch_stale_queued_discarded":
            stale_queued_discarded_by_layer[layer] += 1
            stale_by_layer[layer] += 1
        elif event == "prefetch_stale_queued_discarded_by_worker":
            stale_queued_discarded_by_worker_by_layer[layer] += 1
            stale_by_layer[layer] += 1
        elif event == "prefetch_reading_marked_too_late":
            reading_marked_too_late_by_layer[layer] += 1
            stale_by_layer[layer] += 1
        elif event == "prefetch_too_late_freed":
            too_late_freed_by_layer[layer] += 1

        if event in {
            "prefetch_stale_discarded",
            "prefetch_stale_queued_discarded",
            "prefetch_reading_marked_too_late",
        }:
            # Uno stale successivo dimostra che almeno una richiesta precedente
            # occupava lo slot fino alla domanda reale, anziché essere liberata
            # dal percorso di enqueue.
            if pending_bank_full[layer]:
                bank_full_then_stale_by_layer[layer] += len(pending_bank_full[layer])
                pending_bank_full[layer].clear()

    # Il trace partial_reuse è emesso solo dopo che lo slot è passato da
    # CONSUMING a EMPTY. Un bank-full per il token successivo che lo precede
    # rivela quindi che l'unico slot del layer era ancora in consumo.
    for position, record in enumerate(records):
        if record.get("event") != "prefetch_bank_full":
            continue
        layer = record.get("layer")
        target_token = record.get("token_index")
        if not isinstance(layer, int) or not isinstance(target_token, int):
            continue
        for later in records[position + 1:]:
            if later.get("layer") != layer:
                continue
            if (
                later.get("event") == "prefetch_partial_reuse"
                and isinstance(later.get("token_index"), int)
                and later["token_index"] < target_token
            ):
                bank_full_during_partial_reuse_by_layer[layer] += 1
                break

    all_layers = (
        set(generated_by_layer) | set(queued_by_layer) | set(started_by_layer)
        | set(ready_by_layer) | set(failed_by_layer) | set(bank_full_by_layer)
        | set(stale_by_layer) | set(stale_queued_discarded_by_layer)
        | set(stale_queued_discarded_by_worker_by_layer)
        | set(reading_marked_too_late_by_layer) | set(too_late_freed_by_layer)
        | set(bank_full_during_partial_reuse_by_layer)
    )
    return {
        "records_total": len(records),
        "event_counts": dict(sorted(event_counts.items())),
        "prefetch_bank_full_total": sum(bank_full_by_layer.values()),
        "prefetch_stale_discarded_total": sum(stale_by_layer.values()),
        "prefetch_stale_queued_discarded_total": sum(stale_queued_discarded_by_layer.values()),
        "prefetch_stale_queued_discarded_by_worker_total": sum(
            stale_queued_discarded_by_worker_by_layer.values()
        ),
        "prefetch_reading_marked_too_late_total": sum(reading_marked_too_late_by_layer.values()),
        "prefetch_too_late_freed_total": sum(too_late_freed_by_layer.values()),
        "bank_full_followed_by_stale_discard_total": sum(bank_full_then_stale_by_layer.values()),
        "bank_full_during_previous_partial_reuse_total": sum(
            bank_full_during_partial_reuse_by_layer.values()
        ),
        "per_layer": {
            str(layer): {
                "learned_requests_generated": generated_by_layer[layer],
                "requests_accepted_into_queued": queued_by_layer[layer],
                "worker_reads_started": started_by_layer[layer],
                "worker_reads_ready": ready_by_layer[layer],
                "worker_reads_failed": failed_by_layer[layer],
                "prefetch_bank_full": bank_full_by_layer[layer],
                "prefetch_bank_full_reason": dict(sorted(bank_full_reason_by_layer[layer].items())),
                "prefetch_stale_discarded": stale_by_layer[layer],
                "prefetch_stale_queued_discarded": stale_queued_discarded_by_layer[layer],
                "prefetch_stale_queued_discarded_by_worker": (
                    stale_queued_discarded_by_worker_by_layer[layer]
                ),
                "prefetch_reading_marked_too_late": reading_marked_too_late_by_layer[layer],
                "prefetch_too_late_freed": too_late_freed_by_layer[layer],
                "bank_full_followed_by_stale_discard": bank_full_then_stale_by_layer[layer],
                "bank_full_during_previous_partial_reuse": bank_full_during_partial_reuse_by_layer[layer],
            }
            for layer in sorted(all_layers)
        },
        "worker_and_bank_diagnostics": {
            "snapshot_available": any(
                isinstance(record.get("pinned_bank_snapshot"), dict) for record in records
            ),
            "bank_full_reason_total": dict(sorted(bank_full_reason_total.items())),
            "bank_full_same_layer_state_patterns": dict(sorted(bank_full_slot_states.items())),
            "bank_full_lower_layer_loading_slots": summary(bank_full_lower_loading_slots),
            "bank_full_occupied_slot_age_ms": summary(bank_full_slot_ages),
            "worker_wait_before_read_ms": summary(worker_wait_ms),
            "worker_read_ms": summary(worker_read_ms),
            "worker_queue_to_ready_ms": summary(worker_total_ms),
            "worker_reads_started_by_worker_id": dict(sorted(worker_started_by_id.items())),
            "worker_reads_ready_by_worker_id": dict(sorted(worker_ready_by_id.items())),
            "worker_read_ms_by_worker_id": {
                str(worker_id): summary(values)
                for worker_id, values in sorted(worker_read_ms_by_id.items())
            },
            "bank_loading_slots_observed": summary(loading_slots_observed),
            "lower_layer_loading_slots_observed": summary(lower_layer_loading_observed),
            "pinned_slots_allocated_observed": summary(pinned_slots_observed),
            "pinned_bytes_estimated_observed": summary(pinned_bytes_observed),
        },
        "interpretation": (
            "TEST_9 diagnostico: tutti_slot_queued con molti "
            "lower_layer_loading_slots indica una coda I/O più veloce della "
            "capacità del worker. Con la vecchia scansione per layer questo può "
            "causare starvation dei layer alti; con FIFO mostra invece che le "
            "richieste arrivano più velocemente di quanto i worker possano "
            "leggerle. prefetch_stale_queued_discarded misura gli slot liberati "
            "subito; prefetch_reading_marked_too_late e "
            "prefetch_too_late_freed misurano quelli liberati dal worker dopo "
            "una lettura già iniziata. pinned_bytes_estimated è una stima delle "
            "capacità dei buffer pinned già allocati, non una misura della RAM "
            "fisica disponibile nel sistema. Se la trace contiene worker_id, "
            "i campi per-worker mostrano come le letture siano ripartite fra "
            "i worker abbiano ripartito le letture."
        ),
    }


def build_audit_report(
    trace_path: Path,
    prefetch_layers: str | int = "non specificato",
    prefetch_init_distance: str | int = "non specificato",
) -> dict:
    """Crea il report della politica pinned-RAM per una singola fase C."""
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace prefetch non trovata: {trace_path}")
    return {
        "experiment": "test_9_pinned_ram_policy_audit",
        "trace": str(trace_path),
        "runtime_configuration": {
            "prefetch_layers": str(prefetch_layers),
            "prefetch_init_distance": str(prefetch_init_distance),
        },
        "audit": audit(read_jsonl(trace_path)),
    }


def statistics_summary(values: list[float]) -> dict[str, float | int]:
    """Calcola statistiche descrittive senza dipendenze esterne."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "sample_standard_deviation": 0.0,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def prefetch_worker_count() -> int:
    """Legge e valida il numero di worker SSD richiesti per TEST_9.

    DS4 contiene spazio per al massimo otto worker. Il default è tre,
    perché output_10 ha mostrato che tre letture concorrenti sono il miglior
    compromesso misurato su questa macchina.
    """
    raw_value = os.environ.get("DS4_TEST_9_PREFETCH_NUM_WORKERS", "3")
    try:
        worker_count = int(raw_value)
    except ValueError as error:
        raise SystemExit(
            "DS4_TEST_9_PREFETCH_NUM_WORKERS deve essere un intero tra 1 e 8."
        ) from error
    if worker_count < 1 or worker_count > 8:
        raise SystemExit(
            "DS4_TEST_9_PREFETCH_NUM_WORKERS deve essere compreso tra 1 e 8."
        )
    return worker_count


def benchmark_statistics(repetitions: list[dict]) -> dict:
    """Confronta B e C all'interno della stessa ripetizione."""
    baseline = [float(item["B_replay_baseline"]["wall_seconds"]) for item in repetitions]
    prefetch = [float(item["C_learned_prefetch"]["wall_seconds"]) for item in repetitions]
    deltas = [c - b for b, c in zip(baseline, prefetch)]
    speedups = [(b - c) / b * 100.0 for b, c in zip(baseline, prefetch)]
    return {
        "baseline_wall_seconds": statistics_summary(baseline),
        "learned_prefetch_wall_seconds": statistics_summary(prefetch),
        "prefetch_minus_baseline_seconds": statistics_summary(deltas),
        "speedup_percent": statistics_summary(speedups),
        "pairs_faster_with_prefetch": sum(delta < 0.0 for delta in deltas),
        "pairs_slower_with_prefetch": sum(delta > 0.0 for delta in deltas),
        "pairs_equal": sum(delta == 0.0 for delta in deltas),
    }


def run_c_with_automatic_audit(
    output: Path,
    model: str,
    oracle: Path,
    prefetch_model: str,
    checkpoint: Path,
    layers: str,
    distance: str,
    step: str,
    worker_count: int,
) -> dict:
    """Esegue C e crea subito l'audit TEST_9 accanto alla sua trace."""
    service, socket_path = start_service(output, checkpoint)
    previous_diagnostics = os.environ.get("DS4_TEST_9_PINNED_DIAGNOSTICS")
    previous_parity_workers = os.environ.get("DS4_TEST_9_PARITY_WORKERS")
    previous_worker_policy = os.environ.get("DS4_TEST_9_WORKER_POLICY")
    previous_fifo_workers = os.environ.get("DS4_TEST_9_FIFO_WORKERS")
    previous_partition_workers = os.environ.get("DS4_TEST_9_PARTITION_WORKERS")
    previous_test15_metrics = os.environ.get("DS4_TEST_15_METRICS_PATH")
    # run_ds4 copia l'ambiente del processo. Questa variabile rende più ricca
    # la trace solo per TEST_9; TEST_7 e TEST_8 mantengono il loro formato.
    os.environ["DS4_TEST_9_PINNED_DIAGNOSTICS"] = "1"
    # TEST_7 e TEST_8 non ricevono queste variabili.
    # I worker dividono i layer fisici secondo layer % worker_count e cercano
    # il layer più basso del proprio gruppo.
    os.environ["DS4_TEST_9_PARITY_WORKERS"] = "0"
    os.environ["DS4_TEST_9_WORKER_POLICY"] = "layer_partition"
    os.environ["DS4_TEST_9_FIFO_WORKERS"] = "0"
    os.environ["DS4_TEST_9_PARTITION_WORKERS"] = str(worker_count)
    # TEST_15 usa una cache diversa dalla banca storica a gruppi.  Il CUDA
    # backend scrive qui metriche native della cache, una per fase C, senza
    # cambiare il formato della trace condivisa con TEST_9.
    test15_metrics_path = output / "shared_expert_cache_metrics.json"
    if os.environ.get("DS4_TEST_15_SHARED_EXPERT_CACHE") == "1":
        os.environ["DS4_TEST_15_METRICS_PATH"] = str(test15_metrics_path)
    try:
        result = run_ds4(
            "C", output, model, None, oracle, "prefetch", prefetch_model,
            socket_path, layers, distance, step,
        )
    finally:
        if previous_diagnostics is None:
            os.environ.pop("DS4_TEST_9_PINNED_DIAGNOSTICS", None)
        else:
            os.environ["DS4_TEST_9_PINNED_DIAGNOSTICS"] = previous_diagnostics
        if previous_parity_workers is None:
            os.environ.pop("DS4_TEST_9_PARITY_WORKERS", None)
        else:
            os.environ["DS4_TEST_9_PARITY_WORKERS"] = previous_parity_workers
        if previous_worker_policy is None:
            os.environ.pop("DS4_TEST_9_WORKER_POLICY", None)
        else:
            os.environ["DS4_TEST_9_WORKER_POLICY"] = previous_worker_policy
        if previous_fifo_workers is None:
            os.environ.pop("DS4_TEST_9_FIFO_WORKERS", None)
        else:
            os.environ["DS4_TEST_9_FIFO_WORKERS"] = previous_fifo_workers
        if previous_partition_workers is None:
            os.environ.pop("DS4_TEST_9_PARTITION_WORKERS", None)
        else:
            os.environ["DS4_TEST_9_PARTITION_WORKERS"] = previous_partition_workers
        if previous_test15_metrics is None:
            os.environ.pop("DS4_TEST_15_METRICS_PATH", None)
        else:
            os.environ["DS4_TEST_15_METRICS_PATH"] = previous_test15_metrics
        stop_service(service)

    prefetch_trace = output / "run_C.prefetch.jsonl"
    # Il worker TEST_4 conta anche gli indici fisici 0, 1 e 2, non usati dal
    # predictor learned; per N layer learned deve quindi poter accettare 3+N.
    audit_report = build_audit_report(
        prefetch_trace,
        prefetch_layers=3 + int(layers),
        prefetch_init_distance=distance,
    )
    audit_path = output / "pinned_ram_policy_audit.json"
    write_json(audit_path, audit_report)

    test15_metrics = None
    if test15_metrics_path.is_file():
        with test15_metrics_path.open("r", encoding="utf-8") as handle:
            test15_metrics = json.load(handle)

    return {
        **result,
        **learned_trace_summary(output / "run_C.learned_predictions.jsonl"),
        **prefetch_trace_summary(prefetch_trace, int(layers)),
        **correct_predictions_prefetch_availability_summary(
            output / "run_C.learned_predictions.jsonl", prefetch_trace, int(layers),
        ),
        "pinned_ram_policy_audit_file": str(audit_path),
        "pinned_ram_policy_audit": audit_report["audit"],
        **({
            "shared_expert_cache_metrics_file": str(test15_metrics_path),
            "shared_expert_cache_metrics": test15_metrics,
        } if test15_metrics is not None else {}),
    }


def parse_arguments() -> argparse.Namespace:
    """Seleziona benchmark completo oppure audit manuale di una trace storica."""
    parser = argparse.ArgumentParser(
        description="TEST_9: benchmark A/B/C ripetuto con audit automatico della RAM pinned.",
    )
    parser.add_argument(
        "--repetitions", type=int,
        default=int(os.environ.get("DS4_TEST_9_REPETITIONS", "3")),
        help="Numero di coppie B/C; default: DS4_TEST_9_REPETITIONS oppure 3.",
    )
    parser.add_argument(
        "--audit-trace",
        help="Modalità read-only: analizza solo questa trace C, senza lanciare DS4.",
    )
    arguments = parser.parse_args()
    if not arguments.audit_trace and arguments.repetitions < 2:
        parser.error("--repetitions deve essere almeno 2 per poter alternare l'ordine.")
    return arguments


def audit_only(trace_path: Path) -> None:
    """Mantiene disponibile l'audit manuale di trace create in precedenza."""
    if not trace_path.is_file():
        raise SystemExit(f"Trace prefetch TEST_9 non trovata: {trace_path}")
    output = next_output()
    report = build_audit_report(
        trace_path,
        os.environ.get("DS4_TEST_4_PREFETCH_LAYERS", "non specificato"),
        os.environ.get("DS4_TEST_4_PREFETCH_INIT_DISTANCE", "non specificato"),
    )
    report_path = output / "pinned_ram_policy_audit.json"
    write_json(report_path, report)
    print("TEST_9 audit statico completato. Report:", report_path)


def benchmark_with_automatic_audits(arguments: argparse.Namespace) -> None:
    """Esegue A una volta, poi B/C alternati e audita ogni fase C."""
    model = required_environment("DS4_MODEL")
    prefetch_model = required_environment("DS4_TEST_9_PREFETCH_MODEL")
    layers = os.environ.get("DS4_TEST_9_PREFETCH_LAYERS", "1")
    # Il nuovo nome esplicita che il valore è il numero di slot futuri per
    # ciascun layer. Il vecchio nome resta una compatibilità temporanea per
    # poter rieseguire configurazioni TEST_9 già salvate.
    distance = os.environ.get(
        "DS4_TEST_9_PREFETCH_SLOTS_PER_LAYER",
        os.environ.get("DS4_TEST_9_PREFETCH_INIT_DISTANCE", "1"),
    )
    step = os.environ.get("DS4_TEST_9_PREFETCH_STEP", "1")
    worker_count = prefetch_worker_count()
    checkpoint = Path(os.environ.get("DS4_TEST_9_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint TEST_9 non trovato: {checkpoint}")
    if not Path(model).is_file() or not Path(prefetch_model).is_file():
        raise SystemExit("DS4_MODEL o DS4_TEST_9_PREFETCH_MODEL non punta a un GGUF esistente.")

    output = next_output()
    oracle = output / "routing_oracle.bin"
    report_path = output / "benchmark_summary.json"
    report: dict = {
        "experiment": os.environ.get(
            "DS4_TEST_9_EXPERIMENT", "test_9_repeated_async_learned_prefetch_with_pinned_ram_audit"
        ),
        "status": "running",
        "checkpoint": str(checkpoint),
        "architecture": os.environ.get(
            "DS4_TEST_9_ARCHITECTURE",
            "state512 + hidden512_relu + token64_relu + kv_direct512 + fusion512_relu",
        ),
        "schedule": {
            "layers": int(layers),
            "init_distance": int(distance),
            "step": int(step),
            "prefetch_num_workers": worker_count,
        },
        "worker_policy": (
            f"{worker_count}_ssd_workers_layer_modulo_{worker_count}_low_layer_scan"
        ),
        "repetitions_requested": arguments.repetitions,
        "reference_oracle": {},
        "repetitions": [],
    }
    write_json(report_path, report)
    print(f"TEST_9 output: {output}")
    print("A) Registro una sola volta l'oracolo di routing di riferimento...")
    report["reference_oracle"] = run_ds4(
        "A", output, model, oracle, None, None, None, None, layers, distance, step,
    )
    write_json(report_path, report)

    try:
        for number in range(1, arguments.repetitions + 1):
            order = ("B", "C") if number % 2 else ("C", "B")
            repetition_dir = output / f"repetition_{number:02d}_{'_then_'.join(order)}"
            repetition_dir.mkdir()
            repetition: dict = {"number": number, "order": list(order)}
            print(f"Ripetizione {number}/{arguments.repetitions}: {' poi '.join(order)}...")
            for label in order:
                if label == "B":
                    print(f"  B) Baseline, ripetizione {number}...")
                    repetition["B_replay_baseline"] = run_ds4(
                        "B", repetition_dir, model, None, oracle, None, None, None,
                        layers, distance, step,
                    )
                else:
                    print(f"  C) Learned prefetch asincrono e audit pinned-RAM, ripetizione {number}...")
                    repetition["C_learned_prefetch"] = run_c_with_automatic_audit(
                        repetition_dir, model, oracle, prefetch_model, checkpoint,
                        layers, distance, step, worker_count,
                    )
            baseline_seconds = float(repetition["B_replay_baseline"]["wall_seconds"])
            prefetch_seconds = float(repetition["C_learned_prefetch"]["wall_seconds"])
            repetition["prefetch_minus_baseline_seconds"] = prefetch_seconds - baseline_seconds
            repetition["speedup_percent"] = (baseline_seconds - prefetch_seconds) / baseline_seconds * 100.0
            report["repetitions"].append(repetition)
            report["completed_repetitions"] = len(report["repetitions"])
            report["statistics"] = benchmark_statistics(report["repetitions"])
            write_json(report_path, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["completed_repetitions"] = len(report["repetitions"])
        if report["repetitions"]:
            report["statistics"] = benchmark_statistics(report["repetitions"])
        write_json(report_path, report)
        print("TEST_9 interrotto: risultati delle coppie completate:", report_path)
        raise

    report["status"] = "completed"
    report["completed_repetitions"] = len(report["repetitions"])
    report["statistics"] = benchmark_statistics(report["repetitions"])
    write_json(report_path, report)
    print("TEST_9 completato. Report benchmark:", report_path)


def main() -> None:
    arguments = parse_arguments()
    if arguments.audit_trace:
        audit_only(Path(arguments.audit_trace))
    else:
        benchmark_with_automatic_audits(arguments)


if __name__ == "__main__":
    main()
