#!/usr/bin/env python3
"""Run the isolated TEST_7 A/B/S/C comparison for learned expert prefetching.

Run A records the reference routing oracle.  Run B replays it without any
predictor.  Run S enables the real CPU predictor in shadow mode.  Run C uses
the same predictor output to feed the existing pinned-RAM prefetch worker.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
SERVICE = TEST_DIR / "runtime_predictor_service.py"
BUILD5 = ROOT / "test_predictor" / "prefetch_predictor_pipeline" / "build_dataset_5" / "ml"
DEFAULT_CHECKPOINT = (
    BUILD5 / "neural_network" / "hyperband_fixed_nnz_relu_raw_token_kv_fusion_v1"
    / "trial_checkpoints" / "0024" / "best.weights.h5"
)


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Variabile d'ambiente obbligatoria mancante: {name}")
    return value


def next_output() -> Path:
    for suffix in range(1, 1_000_000):
        directory = TEST_DIR / ("output" if suffix == 1 else f"output_{suffix}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            continue
    raise RuntimeError("Non riesco a creare una nuova directory output TEST_7.")


def generation_tps(stdout_path: Path) -> float:
    matches = re.findall(r"generation:\s*([0-9.]+)\s*t/s", stdout_path.read_text(errors="replace"))
    return float(matches[-1]) if matches else 0.0


def learned_trace_summary(path: Path) -> dict[str, float | int]:
    """Aggregate only records with a known next-token routing target."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    scored = [record for record in records if "recall_at_6" in record]
    if not scored:
        return {"predictions_with_target": 0, "online_recall_at_6": 0.0,
                "predictor_inference_ms_average": 0.0}
    correct_experts = sum(
        len(set(record["predicted_experts"]) & set(record["actual_experts"]))
        for record in scored
    )
    return {
        "predictions_with_target": len(scored),
        "online_recall_at_6": sum(float(record["recall_at_6"]) for record in scored) / len(scored),
        "predictor_inference_ms_average": sum(float(record["inference_ms"]) for record in scored) / len(scored),
        "correct_predicted_experts_total": correct_experts,
        "predicted_experts_total": len(scored) * 6,
        "correct_predicted_experts_average_per_prediction": correct_experts / len(scored),
        "fully_correct_six_expert_predictions": sum(
            len(set(record["predicted_experts"]) & set(record["actual_experts"])) == 6
            for record in scored
        ),
    }


def correct_predictions_prefetch_availability_summary(
    learned_path: Path, prefetch_path: Path, learned_layer_count: int,
) -> dict[str, int]:
    """Spiega quanti expert predetti correttamente erano davvero riusabili.

    Un expert puo' essere corretto rispetto al router del token successivo ma
    non essere ancora presente nella RAM pinned quando DS4 lo domanda. Questa
    funzione unisce la trace del predictor con quella del worker usando la
    chiave ``(target_token_index, layer)`` e separa i motivi osservabili.
    """
    learned_records = [
        json.loads(line) for line in learned_path.read_text().splitlines() if line.strip()
    ]
    prefetch_records = [
        json.loads(line) for line in prefetch_path.read_text().splitlines() if line.strip()
    ]
    predicted_layers = set(range(3, 3 + learned_layer_count))
    demand_events = {"prefetch_hit", "prefetch_partial_reuse", "prefetch_miss_demand"}

    # Conserva la posizione nella trace: per essere utile, worker_ready deve
    # comparire prima della domanda degli expert dello stesso token/layer.
    events_by_key: dict[tuple[int, int], list[tuple[int, dict]]] = {}
    demands_by_key: dict[tuple[int, int], tuple[int, dict]] = {}
    for position, record in enumerate(prefetch_records):
        token_index = record.get("token_index")
        layer = record.get("layer")
        if not isinstance(token_index, int) or layer not in predicted_layers:
            continue
        key = (token_index, layer)
        events_by_key.setdefault(key, []).append((position, record))
        if record.get("event") in demand_events:
            demands_by_key[key] = (position, record)

    result = {
        "correct_predicted_experts_reused_from_pinned_ram": 0,
        "correct_predicted_experts_not_reused": 0,
        "correct_predicted_experts_not_reused_prefetch_bank_full": 0,
        "correct_predicted_experts_not_reused_worker_not_ready": 0,
        "correct_predicted_experts_not_reused_stale_discarded": 0,
        "correct_predicted_experts_not_reused_ready_but_unavailable": 0,
        "correct_predicted_experts_not_reused_no_prefetch_state": 0,
    }
    for prediction in learned_records:
        if "recall_at_6" not in prediction:
            continue
        layer = prediction.get("layer")
        target_token_index = prediction.get("target_token_index")
        if layer not in predicted_layers or not isinstance(target_token_index, int):
            continue
        correct_count = len(
            set(prediction["predicted_experts"]) & set(prediction["actual_experts"])
        )
        if not correct_count:
            continue
        key = (target_token_index, layer)
        demand = demands_by_key.get(key)
        if demand is None:
            result["correct_predicted_experts_not_reused"] += correct_count
            result["correct_predicted_experts_not_reused_no_prefetch_state"] += correct_count
            continue

        demand_position, demand_record = demand
        reused_count = min(correct_count, int(demand_record.get("prefetched_experts_reused", 0)))
        result["correct_predicted_experts_reused_from_pinned_ram"] += reused_count
        not_reused = correct_count - reused_count
        if not not_reused:
            continue
        result["correct_predicted_experts_not_reused"] += not_reused

        earlier_events = {
            record.get("event")
            for position, record in events_by_key.get(key, [])
            if position <= demand_position
        }
        # La priorita' evita doppio conteggio: un gruppo rifiutato per banca
        # piena non puo' essere interpretato come una previsione arrivata tardi.
        if "prefetch_bank_full" in earlier_events:
            reason = "correct_predicted_experts_not_reused_prefetch_bank_full"
        elif "prefetch_stale_discarded" in earlier_events:
            reason = "correct_predicted_experts_not_reused_stale_discarded"
        elif (
            "prefetch_worker_queued" in earlier_events
            or "prefetch_worker_started" in earlier_events
        ) and "prefetch_worker_ready" not in earlier_events:
            reason = "correct_predicted_experts_not_reused_worker_not_ready"
        elif "prefetch_worker_ready" in earlier_events:
            reason = "correct_predicted_experts_not_reused_ready_but_unavailable"
        else:
            reason = "correct_predicted_experts_not_reused_no_prefetch_state"
        result[reason] += not_reused
    return result


def prefetch_trace_summary(path: Path, learned_layer_count: int) -> dict[str, int]:
    """Summarize only demand loads for the MoE layers predicted by TEST_7."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    predicted_layers = set(range(3, 3 + learned_layer_count))
    demand_events = {"prefetch_hit", "prefetch_partial_reuse", "prefetch_miss_demand"}
    relevant_demands = [
        record for record in records
        if record.get("event") in demand_events and record.get("layer") in predicted_layers
    ]
    return {
        "learned_prefetch_enqueues": sum(
            record.get("event") == "learned_prefetch_enqueue" and record.get("layer") in predicted_layers
            for record in records
        ),
        "expert_group_demands_at_predicted_layers": len(relevant_demands),
        "full_prefetch_hits": sum(record.get("event") == "prefetch_hit" for record in relevant_demands),
        "partial_prefetch_reuse_groups": sum(
            record.get("event") == "prefetch_partial_reuse" for record in relevant_demands
        ),
        "experts_reused_from_pinned_ram": sum(
            int(record.get("prefetched_experts_reused", 0)) for record in relevant_demands
        ),
        "experts_loaded_from_ssd_at_predicted_layers": sum(
            int(record.get("ssd_experts_loaded", 0)) for record in relevant_demands
        ),
    }


def run_ds4(label: str, output: Path, model: str, oracle_write: Path | None,
            oracle_replay: Path | None, mode: str | None, prefetch_model: str | None,
            socket_path: Path | None, layers: str, distance: str, step: str) -> dict[str, float | int]:
    """Execute exactly one DS4 subprocess and return runner-side measurements."""
    environment = os.environ.copy()
    for variable in (
        "DS4_ROUTING_ORACLE_WRITE_PATH", "DS4_ROUTING_ORACLE_REPLAY_PATH",
        "DS4_ROUTING_TRACE_PATH", "DS4_TEST_4_TRACE_PATH", "DS4_TEST_4_PREFETCH",
        "DS4_TEST_4_PREFETCH_LAYERS", "DS4_TEST_4_PREFETCH_INIT_DISTANCE",
        "DS4_TEST_4_PREFETCH_STEP", "DS4_TEST_5_PREFETCH_MODEL",
        "DS4_TEST_7_MODE", "DS4_TEST_7_SOCKET_PATH", "DS4_TEST_7_TRACE_PATH",
    ):
        environment.pop(variable, None)
    environment["DS4_CUDA_MMQ"] = "0"
    environment["DS4_ROUTING_TRACE_PATH"] = str(output / f"run_{label}.routing.jsonl")
    if oracle_write:
        environment["DS4_ROUTING_ORACLE_WRITE_PATH"] = str(oracle_write)
    if oracle_replay:
        environment["DS4_ROUTING_ORACLE_REPLAY_PATH"] = str(oracle_replay)
    if mode:
        environment["DS4_TEST_7_MODE"] = mode
        environment["DS4_TEST_7_SOCKET_PATH"] = str(socket_path)
        environment["DS4_TEST_7_TRACE_PATH"] = str(output / f"run_{label}.learned_predictions.jsonl")
        # DS4 legge direttamente questa variabile per decidere su quanti layer
        # MoE inviare richieste al predictor. Impostarla esplicitamente qui
        # evita che un runner che riusa questa funzione (TEST_8) cada nel
        # valore predefinito di un solo layer.
        environment["DS4_TEST_7_PREFETCH_LAYERS"] = layers
    if prefetch_model:
        environment["DS4_TEST_4_PREFETCH"] = "1"
        environment["DS4_TEST_4_TRACE_PATH"] = str(output / f"run_{label}.prefetch.jsonl")
        # TEST_7 interpreta ``layers`` come numero di layer MoE da predire,
        # a partire dal primo layer MoE fisico (indice 3). Il worker pinned-RAM
        # storico invece riceve un numero di indici fisici contati da zero:
        # con 1 layer TEST_7 deve quindi poter accettare almeno gli indici 0..3.
        # Gli indici 0..2 non ricevono prefetch learned; riservarli evita soltanto
        # che il worker rifiuti il primo enqueue reale al layer fisico 3.
        environment["DS4_TEST_4_PREFETCH_LAYERS"] = str(3 + int(layers))
        environment["DS4_TEST_4_PREFETCH_INIT_DISTANCE"] = distance
        environment["DS4_TEST_4_PREFETCH_STEP"] = step
        environment["DS4_TEST_5_PREFETCH_MODEL"] = prefetch_model

    command = [
        str(ROOT / "ds4"), "--cuda", "--ssd-streaming", "--ssd-streaming-cache-experts", "8GB",
        "-c", "1024", "-n", "64", "--nothink", "--temp", "0", "-p",
        "Explain what a binary search is in one short paragraph.", "-m", model,
    ]
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    (output / f"run_{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / f"run_{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Run {label} fallita con codice {result.returncode}; vedere stderr.")
    return {"wall_seconds": elapsed, "generation_tokens_per_second": generation_tps(output / f"run_{label}.stdout.txt")}


def start_service(output: Path, checkpoint: Path) -> tuple[subprocess.Popen[str], Path]:
    socket_path = output / "predictor.sock"
    command = [
        sys.executable, str(SERVICE), "--socket-path", str(socket_path),
        "--checkpoint", str(checkpoint),
        "--state-artifacts", str(BUILD5 / "ablations" / "09_routing_layer_router_state_kv_current" / "artifacts"),
        "--hidden-artifacts", str(BUILD5 / "ablations" / "11_routing_layer_router_state_kv_current_raw_hidden_token_embedding" / "artifacts"),
        "--kv-artifacts", str(BUILD5 / "ablations" / "14_routing_layer_router_state_kv_current_materialized_kv_f16" / "artifacts"),
        "--token-data-dir", str(BUILD5 / "ablations" / "10_routing_layer_router_state_kv_current_token_embedding"),
        "--token-embedding-dir", str(BUILD5 / "token_embedding_artifacts"),
    ]
    environment = os.environ.copy()
    # TensorFlow deve restare rigorosamente sulla CPU, anche prima che il
    # servizio chiami tf.config.set_visible_devices([], "GPU").
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    log_path = output / "predictor_service.log"

    # Un primo avvio TensorFlow può terminare transitoriamente mentre CUDA è
    # appena stata usata da DS4 nella fase B. Un solo secondo tentativo rende
    # il test ripetibile e il log conserva entrambi gli esiti.
    for attempt in range(1, 3):
        socket_path.unlink(missing_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            print(f"TEST_7 runner: avvio servizio CPU, tentativo {attempt}/2.", file=log, flush=True)
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        deadline = time.monotonic() + 180.0
        while not socket_path.exists():
            return_code = process.poll()
            if return_code is not None:
                if attempt == 2:
                    raise RuntimeError(
                        "Il servizio predictor si è arrestato "
                        f"(codice {return_code}); vedere predictor_service.log."
                    )
                break
            if time.monotonic() >= deadline:
                process.terminate()
                raise RuntimeError("Timeout nel caricamento del servizio predictor.")
            time.sleep(0.1)
        else:
            return process, socket_path
        time.sleep(1.0)

    raise AssertionError("Avvio predictor non raggiungibile")


def stop_service(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    model = required_environment("DS4_MODEL")
    prefetch_model = required_environment("DS4_TEST_7_PREFETCH_MODEL")
    layers = os.environ.get("DS4_TEST_7_PREFETCH_LAYERS", "5")
    distance = os.environ.get("DS4_TEST_7_PREFETCH_INIT_DISTANCE", "1")
    step = os.environ.get("DS4_TEST_7_PREFETCH_STEP", "1")
    checkpoint = Path(os.environ.get("DS4_TEST_7_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint TEST_7 non trovato: {checkpoint}")
    if not Path(model).is_file() or not Path(prefetch_model).is_file():
        raise SystemExit("DS4_MODEL o DS4_TEST_7_PREFETCH_MODEL non punta a un GGUF esistente.")

    output = next_output()
    oracle = output / "routing_oracle.bin"
    print(f"TEST_7 output: {output}")
    print("A) Registro l'oracolo di routing di riferimento...")
    run_a = run_ds4("A", output, model, oracle, None, None, None, None, layers, distance, step)
    print("B) Replay di controllo, senza predictor né prefetch...")
    run_b = run_ds4("B", output, model, None, oracle, None, None, None, layers, distance, step)

    print("S) Fase shadow: predictor CPU reale, senza prefetch...")
    service, socket_path = start_service(output, checkpoint)
    try:
        run_s = run_ds4("S", output, model, None, oracle, "shadow", None, socket_path, layers, distance, step)
    finally:
        stop_service(service)

    print("C) Fase learned prefetch: le sei predizioni alimentano il worker pinned-RAM...")
    service, socket_path = start_service(output, checkpoint)
    try:
        run_c = run_ds4("C", output, model, None, oracle, "prefetch", prefetch_model, socket_path, layers, distance, step)
    finally:
        stop_service(service)

    report = {
        "experiment": "test_7_build5_learned_cpu_prefetch",
        "checkpoint": str(checkpoint),
        "architecture": "state512 + hidden512_relu + token64_relu + kv_direct512 + fusion512_relu",
        "phases": {"A_oracle_reference": run_a, "B_replay_baseline": run_b,
                   "S_shadow_predictor": {**run_s, **learned_trace_summary(output / "run_S.learned_predictions.jsonl")},
                   "C_learned_prefetch": {
                       **run_c,
                       **learned_trace_summary(output / "run_C.learned_predictions.jsonl"),
                       **prefetch_trace_summary(output / "run_C.prefetch.jsonl", int(layers)),
                       **correct_predictions_prefetch_availability_summary(
                           output / "run_C.learned_predictions.jsonl",
                           output / "run_C.prefetch.jsonl",
                           int(layers),
                       ),
                   }},
        "schedule": {"layers": int(layers), "init_distance": int(distance), "step": int(step)},
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("TEST_7 completato. Report:", output / "summary.json")


if __name__ == "__main__":
    main()
