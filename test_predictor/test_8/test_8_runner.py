#!/usr/bin/env python3
"""Benchmark ripetuto B/C per misurare TEST_7 in modo meno sensibile alle cache.

TEST_8 non modifica DS4 e non introduce un nuovo predictor: riusa esattamente
il servizio CPU e il percorso asincrono verificati in TEST_7. Registra un solo
oracolo e poi alterna l'ordine delle fasi B e C per ogni ripetizione.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
TEST7_DIR = ROOT / "test_predictor" / "test_7"
sys.path.insert(0, str(TEST7_DIR))

# TEST_7 rimane l'unico posto che implementa protocollo socket, servizio
# TensorFlow CPU e chiamata DS4. TEST_8 aggiunge soltanto il protocollo di
# benchmark ripetuto e l'analisi statistica.
from test_7_runner import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    learned_trace_summary,
    prefetch_trace_summary,
    required_environment,
    run_ds4,
    start_service,
    stop_service,
)


def next_output() -> Path:
    """Crea una directory TEST_8 nuova senza sovrascrivere risultati precedenti."""
    for suffix in range(1, 1_000_000):
        directory = TEST_DIR / ("output" if suffix == 1 else f"output_{suffix}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            continue
    raise RuntimeError("Non riesco a creare una nuova directory output TEST_8.")


def write_json(path: Path, value: dict) -> None:
    """Salva in modo atomico anche il progresso disponibile dopo Ctrl+C."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def statistics_summary(values: list[float]) -> dict[str, float | int]:
    """Restituisce le statistiche descrittive senza richiedere librerie esterne."""
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "minimum": 0.0, "maximum": 0.0, "sample_standard_deviation": 0.0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def benchmark_summary(repetitions: list[dict]) -> dict:
    """Calcola confronti accoppiati: ogni B viene confrontata con la C della stessa ripetizione."""
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


def run_c(output: Path, model: str, oracle: Path, prefetch_model: str,
          checkpoint: Path, layers: str, distance: str, step: str) -> dict:
    """Avvia il servizio CPU solo per la fase C e ne salva trace e contatori."""
    service, socket_path = start_service(output, checkpoint)
    try:
        result = run_ds4(
            "C", output, model, None, oracle, "prefetch", prefetch_model,
            socket_path, layers, distance, step,
        )
    finally:
        stop_service(service)
    return {
        **result,
        **learned_trace_summary(output / "run_C.learned_predictions.jsonl"),
        **prefetch_trace_summary(output / "run_C.prefetch.jsonl", int(layers)),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TEST_8: benchmark B/C ripetuto con ordine alternato.",
    )
    parser.add_argument(
        "--repetitions", type=int,
        default=int(os.environ.get("DS4_TEST_8_REPETITIONS", "3")),
        help="Numero di coppie B/C; default: DS4_TEST_8_REPETITIONS oppure 3.",
    )
    arguments = parser.parse_args()
    if arguments.repetitions < 2:
        parser.error("--repetitions deve essere almeno 2 per poter alternare l'ordine.")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    model = required_environment("DS4_MODEL")
    prefetch_model = required_environment("DS4_TEST_8_PREFETCH_MODEL")
    layers = os.environ.get("DS4_TEST_8_PREFETCH_LAYERS", "1")
    distance = os.environ.get("DS4_TEST_8_PREFETCH_INIT_DISTANCE", "1")
    step = os.environ.get("DS4_TEST_8_PREFETCH_STEP", "1")
    checkpoint = Path(os.environ.get("DS4_TEST_8_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint TEST_8 non trovato: {checkpoint}")
    if not Path(model).is_file() or not Path(prefetch_model).is_file():
        raise SystemExit("DS4_MODEL o DS4_TEST_8_PREFETCH_MODEL non punta a un GGUF esistente.")

    output = next_output()
    oracle = output / "routing_oracle.bin"
    report_path = output / "benchmark_summary.json"
    report: dict = {
        "experiment": "test_8_repeated_async_learned_prefetch_benchmark",
        "status": "running",
        "checkpoint": str(checkpoint),
        "architecture": "state512 + hidden512_relu + token64_relu + kv_direct512 + fusion512_relu",
        "schedule": {"layers": int(layers), "init_distance": int(distance), "step": int(step)},
        "repetitions_requested": arguments.repetitions,
        "reference_oracle": {},
        "repetitions": [],
    }
    write_json(report_path, report)
    print(f"TEST_8 output: {output}")
    print("A) Registro una sola volta l'oracolo di routing di riferimento...")
    report["reference_oracle"] = run_ds4(
        "A", output, model, oracle, None, None, None, None, layers, distance, step,
    )
    write_json(report_path, report)

    try:
        for number in range(1, arguments.repetitions + 1):
            # B-C, C-B, B-C, ...: su un numero dispari B e C differiscono al
            # massimo di una posizione nell'ordine, evitando che C sia sempre
            # favorita o sfavorita dalle cache gia' calde.
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
                    print(f"  C) Learned prefetch asincrono, ripetizione {number}...")
                    repetition["C_learned_prefetch"] = run_c(
                        repetition_dir, model, oracle, prefetch_model, checkpoint,
                        layers, distance, step,
                    )
            baseline_seconds = float(repetition["B_replay_baseline"]["wall_seconds"])
            prefetch_seconds = float(repetition["C_learned_prefetch"]["wall_seconds"])
            repetition["prefetch_minus_baseline_seconds"] = prefetch_seconds - baseline_seconds
            repetition["speedup_percent"] = (baseline_seconds - prefetch_seconds) / baseline_seconds * 100.0
            report["repetitions"].append(repetition)
            report["completed_repetitions"] = len(report["repetitions"])
            report["statistics"] = benchmark_summary(report["repetitions"])
            # Ogni coppia completa e' consultabile anche se l'utente interrompe
            # il benchmark durante una ripetizione successiva.
            write_json(report_path, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["completed_repetitions"] = len(report["repetitions"])
        if report["repetitions"]:
            report["statistics"] = benchmark_summary(report["repetitions"])
        write_json(report_path, report)
        print("TEST_8 interrotto: risultati delle coppie completate:", report_path)
        raise

    report["status"] = "completed"
    report["completed_repetitions"] = len(report["repetitions"])
    report["statistics"] = benchmark_summary(report["repetitions"])
    write_json(report_path, report)
    print("TEST_8 completato. Report:", report_path)


if __name__ == "__main__":
    main()
