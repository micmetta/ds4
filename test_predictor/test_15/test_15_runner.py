#!/usr/bin/env python3
"""TEST_15: misura non invasiva dell'overlap fra slot pinned di TEST_14."""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "test_predictor" / "test_9"))
sys.path.insert(0, str(ROOT / "test_predictor" / "test_14"))
import test_9_runner as test9  # noqa: E402
import test_14_runner as test14  # noqa: E402


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Variabile obbligatoria mancante: {name}")
    return value


def configure_record_environment() -> tuple[str, str, str, str, int, Path]:
    """Mappa l'interfaccia TEST_15 sulla collaudata infrastruttura TEST_9/14."""
    checkpoint = Path(required("DS4_TEST_15_CHECKPOINT"))
    top_k = int(os.environ.get("DS4_TEST_15_PREFETCH_TOP_K", "6"))
    if top_k != 6:
        raise SystemExit("TEST_15 misura la configurazione record: DS4_TEST_15_PREFETCH_TOP_K deve essere 6.")
    workers = int(os.environ.get("DS4_TEST_15_PREFETCH_NUM_WORKERS", "6"))
    if workers != 6:
        raise SystemExit("TEST_15 misura la configurazione record: DS4_TEST_15_PREFETCH_NUM_WORKERS deve essere 6.")
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint TEST_15 non trovato: {checkpoint}")

    model = required("DS4_MODEL")
    prefetch_model = required("DS4_TEST_15_PREFETCH_MODEL")
    if not Path(model).is_file() or not Path(prefetch_model).is_file():
        raise SystemExit("DS4_MODEL o DS4_TEST_15_PREFETCH_MODEL non punta a un GGUF esistente.")

    layers = "40"
    distance = os.environ.get("DS4_TEST_15_PREFETCH_SLOTS_PER_LAYER", "2")
    step = os.environ.get("DS4_TEST_15_PREFETCH_STEP", "1")
    if distance != "2" or step != "1":
        raise SystemExit("TEST_15 richiede due slot futuri per layer e step=1, come TEST_14/output_2.")

    os.environ.update({
        "DS4_TEST_9_PREFETCH_MODEL": prefetch_model,
        "DS4_TEST_9_CHECKPOINT": str(checkpoint),
        "DS4_TEST_9_PREFETCH_LAYERS": layers,
        "DS4_TEST_9_PREFETCH_SLOTS_PER_LAYER": distance,
        "DS4_TEST_9_PREFETCH_STEP": step,
        "DS4_TEST_9_PREFETCH_NUM_WORKERS": str(workers),
        "DS4_TEST_14_PREFETCH_TOP_K": str(top_k),
        "DS4_TEST_13_WINDOWS_BRIDGE": "1",
        "DS4_TEST_13_WINDOWS_HOST": required("DS4_TEST_15_WINDOWS_HOST"),
        "DS4_TEST_13_WINDOWS_PORT": os.environ.get("DS4_TEST_15_WINDOWS_PORT", "38114"),
    })
    return model, prefetch_model, layers, distance, workers, checkpoint


def overlap_report(trace_path: Path) -> dict:
    """Quantifica gli expert sovrapposti negli altri slot del medesimo layer.

    La funzione e' volutamente read-only: usa solo gli snapshot prodotti al
    momento dell'enqueue e non influenza la scelta dello slot o l'I/O.
    """
    records = test9.read_jsonl(trace_path)
    states = Counter()
    per_layer: dict[int, Counter] = defaultdict(Counter)
    requests = 0
    snapshots = 0
    overlap_pairs = 0
    overlap_experts = 0
    immediate_ready_pairs = 0
    immediate_ready_experts = 0
    requested_bytes = 0.0
    immediately_reusable_bytes = 0.0
    missing_expert_snapshots = 0

    for record in records:
        # Lo snapshot atomico e' emesso dal CUDA worker come evento QUEUED,
        # subito prima del record learned_prefetch_enqueue del thread DS4.
        # Quest'ultimo non porta volutamente una copia della banca pinned.
        if record.get("event") != "prefetch_worker_queued":
            continue
        experts = {int(value) for value in record.get("experts", []) if isinstance(value, int) and value >= 0}
        if not experts:
            continue
        requests += 1
        layer = int(record.get("layer", -1))
        requested_bytes += float(record.get("bytes", 0.0))
        snapshot = record.get("pinned_bank_snapshot")
        if not isinstance(snapshot, dict):
            missing_expert_snapshots += 1
            continue
        own_distance = snapshot.get("slot_distance_index")
        slots = snapshot.get("same_layer_slots")
        if not isinstance(slots, list):
            missing_expert_snapshots += 1
            continue
        for slot in slots:
            if not isinstance(slot, dict) or slot.get("distance_index") == own_distance:
                continue
            slot_experts = {int(value) for value in slot.get("experts", []) if isinstance(value, int) and value >= 0}
            if not slot_experts:
                continue
            snapshots += 1
            state = str(slot.get("state", "UNKNOWN"))
            states[state] += 1
            per_layer[layer]["sibling_slot_observations"] += 1
            common = experts & slot_experts
            if not common:
                continue
            overlap_pairs += 1
            overlap_experts += len(common)
            per_layer[layer]["overlap_pairs"] += 1
            per_layer[layer]["overlap_experts"] += len(common)
            if state == "READY":
                immediate_ready_pairs += 1
                immediate_ready_experts += len(common)
                per_layer[layer]["ready_overlap_pairs"] += 1
                per_layer[layer]["ready_overlap_experts"] += len(common)
                bytes_per_expert = float(record.get("bytes", 0.0)) / len(experts)
                immediately_reusable_bytes += len(common) * bytes_per_expert

    return {
        "trace": str(trace_path),
        "measurement_scope": (
            "snapshot atomico dell'evento prefetch_worker_queued; nessuna modifica "
            "alla politica di selezione, lettura o riuso degli slot"
        ),
        "learned_prefetch_requests": requests,
        "sibling_slot_observations_with_expert_lists": snapshots,
        "requests_without_usable_snapshot": missing_expert_snapshots,
        "sibling_slot_states": dict(sorted(states.items())),
        "overlap_pairs": overlap_pairs,
        "overlap_experts": overlap_experts,
        "overlap_experts_per_sibling_observation": overlap_experts / snapshots if snapshots else 0.0,
        "ready_overlap_pairs": immediate_ready_pairs,
        "ready_overlap_experts": immediate_ready_experts,
        "ready_overlap_experts_per_request": immediate_ready_experts / requests if requests else 0.0,
        "requested_bytes": requested_bytes,
        "bytes_immediately_reusable_if_composed": immediately_reusable_bytes,
        "immediately_reusable_byte_fraction": (
            immediately_reusable_bytes / requested_bytes if requested_bytes else 0.0
        ),
        "per_layer": {str(layer): dict(sorted(values.items())) for layer, values in sorted(per_layer.items())},
    }


def aggregate_overlap(repetitions: list[dict]) -> dict:
    keys = (
        "learned_prefetch_requests", "sibling_slot_observations_with_expert_lists",
        "overlap_pairs", "overlap_experts", "ready_overlap_pairs", "ready_overlap_experts",
        "requested_bytes", "bytes_immediately_reusable_if_composed",
    )
    totals = {key: sum(float(item["overlap"][key]) for item in repetitions) for key in keys}
    return {
        "repetitions": len(repetitions),
        "totals": totals,
        "ready_overlap_experts_per_request": (
            totals["ready_overlap_experts"] / totals["learned_prefetch_requests"]
            if totals["learned_prefetch_requests"] else 0.0
        ),
        "immediately_reusable_byte_fraction": (
            totals["bytes_immediately_reusable_if_composed"] / totals["requested_bytes"]
            if totals["requested_bytes"] else 0.0
        ),
        "interpretation": (
            "Solo l'overlap osservato in uno slot READY e' immediatamente riusabile "
            "senza attendere altra I/O. Questo report misura un potenziale: non prova "
            "che la composizione fra slot o una cache per expert producano speedup end-to-end."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TEST_15: profiling read-only dell'overlap nella banca pinned.")
    parser.add_argument("--repetitions", type=int, default=int(os.environ.get("DS4_TEST_15_REPETITIONS", "3")))
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        raise SystemExit("--repetitions deve essere positivo.")

    model, prefetch_model, layers, distance, workers, checkpoint = configure_record_environment()
    test9.TEST_DIR = TEST_DIR
    test9.start_service = test14.start_build7_service
    test9.run_c_with_automatic_audit = test14.run_c_build7
    output = test9.next_output()
    oracle = output / "routing_oracle.bin"
    report_path = output / "overlap_measurement_report.json"
    report: dict = {
        "experiment": "test_15_build7_top6_six_workers_pinned_slot_overlap_measurement",
        "status": "running",
        "configuration": {
            "source_record": "TEST_14/output_2",
            "checkpoint": str(checkpoint),
            "prefetch_top_k": 6,
            "learned_physical_layers": "5-42",
            "worker_window": "3-42",
            "slots_per_layer": 2,
            "workers": workers,
        },
        "reference_oracle": {},
        "repetitions": [],
    }
    test9.write_json(report_path, report)
    print(f"TEST_15 output: {output}")
    print("A) Registro una sola volta l'oracolo di routing necessario alle sole fasi C...")
    report["reference_oracle"] = test9.run_ds4(
        "A", output, model, oracle, None, None, None, None, layers, distance, "1",
    )
    test9.write_json(report_path, report)

    try:
        for number in range(1, arguments.repetitions + 1):
            repetition_dir = output / f"repetition_{number:02d}_C_only"
            repetition_dir.mkdir()
            print(f"C) Learned prefetch record, ripetizione {number}/{arguments.repetitions}...")
            result = test14.run_c_build7(
                repetition_dir, model, oracle, prefetch_model, checkpoint,
                layers, distance, "1", workers,
            )
            overlap = overlap_report(repetition_dir / "run_C.prefetch.jsonl")
            repetition = {"number": number, "C_learned_prefetch": result, "overlap": overlap}
            report["repetitions"].append(repetition)
            report["completed_repetitions"] = len(report["repetitions"])
            report["aggregate_overlap"] = aggregate_overlap(report["repetitions"])
            test9.write_json(report_path, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["completed_repetitions"] = len(report["repetitions"])
        if report["repetitions"]:
            report["aggregate_overlap"] = aggregate_overlap(report["repetitions"])
        test9.write_json(report_path, report)
        print("TEST_15 interrotto: report parziale conservato in", report_path)
        raise

    report["status"] = "completed"
    report["completed_repetitions"] = len(report["repetitions"])
    report["aggregate_overlap"] = aggregate_overlap(report["repetitions"])
    test9.write_json(report_path, report)
    print("TEST_15 completato. Report overlap:", report_path)


if __name__ == "__main__":
    main()
