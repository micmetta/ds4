#!/usr/bin/env python3
"""TEST_14: repeat TEST_13/output_9 with the selected Build-7 Top-4 model."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
SERVICE = TEST_DIR / "runtime_predictor_build7_service.py"
BUILD7 = ROOT / "test_predictor" / "prefetch_predictor_pipeline" / "build_dataset_7" / "ml"
sys.path.insert(0, str(ROOT / "test_predictor" / "test_9"))
import test_9_runner as test9  # noqa: E402


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Variabile obbligatoria mancante: {name}")
    return value


def start_build7_service(output: Path, checkpoint: Path) -> tuple[subprocess.Popen[str], Path]:
    """Start the CPU-only service with exactly the artifacts of the chosen model."""
    socket_path = output / "predictor.sock"
    command = [
        sys.executable, str(SERVICE), "--socket-path", str(socket_path),
        "--checkpoint", str(checkpoint),
        "--prefetch-top-k", os.environ["DS4_TEST_14_PREFETCH_TOP_K"],
        "--state-artifacts", str(BUILD7 / "ablations" / "01_current_previous_previous_state" / "artifacts"),
        "--fixed-nnz-artifacts", str(BUILD7 / "ablations" / "11_current_fixed_nnz_state" / "artifacts"),
        "--current-hidden-artifacts", str(BUILD7 / "ablations" / "03_current_hidden_refs" / "artifacts"),
        "--previous-hidden-artifacts", str(BUILD7 / "ablations" / "06_previous_hidden_refs" / "artifacts"),
        "--transition-artifacts", str(BUILD7 / "transition_features" / "expert_transition_probabilities_v1" / "artifacts"),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    log_path = output / "predictor_service.log"
    for attempt in range(1, 3):
        socket_path.unlink(missing_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            print(f"TEST_14 runner: avvio servizio CPU, tentativo {attempt}/2.", file=log, flush=True)
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 180.0
        while not socket_path.exists():
            if process.poll() is not None:
                if attempt == 2:
                    raise RuntimeError("Il servizio Build 7 si è arrestato; vedere predictor_service.log.")
                break
            if time.monotonic() >= deadline:
                process.terminate()
                raise RuntimeError("Timeout nel caricamento del servizio Build 7.")
            time.sleep(0.1)
        else:
            return process, socket_path
        time.sleep(1.0)
    raise AssertionError("Avvio servizio TEST_14 non raggiungibile")


def run_c_build7(*args, **kwargs):
    """Enable protocol T7P3 only for phase C; A/B remain pure reference runs."""
    previous = os.environ.get("DS4_TEST_14_ENABLED")
    os.environ["DS4_TEST_14_ENABLED"] = "1"
    try:
        return original_run_c(*args, **kwargs)
    finally:
        if previous is None:
            os.environ.pop("DS4_TEST_14_ENABLED", None)
        else:
            os.environ["DS4_TEST_14_ENABLED"] = previous


original_run_c = test9.run_c_with_automatic_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="TEST_14: bridge TEST_13 con predictor Build 7 Top-4.")
    parser.add_argument("--repetitions", type=int, default=int(os.environ.get("DS4_TEST_14_REPETITIONS", "3")))
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions deve essere positivo.")

    checkpoint = required("DS4_TEST_14_CHECKPOINT")
    prefetch_top_k = int(os.environ.get("DS4_TEST_14_PREFETCH_TOP_K", "6"))
    if not 6 <= prefetch_top_k <= 20:
        raise SystemExit("DS4_TEST_14_PREFETCH_TOP_K deve essere un intero tra 6 e 20.")
    # TEST_9's orchestration retains the physical window 3--42 (40 layers),
    # so its pinned worker accepts layer 42.  Build 7 itself sends requests
    # only for its valid learned interval 5--42 (38 layers).
    os.environ.update({
        "DS4_TEST_9_PREFETCH_MODEL": required("DS4_TEST_14_PREFETCH_MODEL"),
        "DS4_TEST_9_CHECKPOINT": checkpoint,
        "DS4_TEST_9_PREFETCH_LAYERS": "40",
        "DS4_TEST_9_PREFETCH_SLOTS_PER_LAYER": os.environ.get("DS4_TEST_14_PREFETCH_SLOTS_PER_LAYER", "2"),
        "DS4_TEST_9_PREFETCH_STEP": os.environ.get("DS4_TEST_14_PREFETCH_STEP", "1"),
        "DS4_TEST_9_PREFETCH_NUM_WORKERS": os.environ.get("DS4_TEST_14_PREFETCH_NUM_WORKERS", "6"),
        "DS4_TEST_14_PREFETCH_TOP_K": str(prefetch_top_k),
        "DS4_TEST_9_EXPERIMENT": "test_14_build7_top4_async_learned_prefetch",
        "DS4_TEST_9_ARCHITECTURE": (
            "Build7 Top-4: fixed-NNZ state2048 + raw_hidden_l64 + raw_hidden_l_minus_1_64 "
            "+ q256 gated residual + fusion2048; learned physical layers 5-42; "
            f"prefetch Top-{prefetch_top_k}"
        ),
    })
    # Keep the existing DS4 bridge namespace internally while exposing a clean
    # TEST_14-facing configuration to the user.  The TEST_14 Windows reader
    # accepts the requested Top-K in the native request header.
    os.environ["DS4_TEST_13_WINDOWS_BRIDGE"] = "1"
    os.environ["DS4_TEST_13_WINDOWS_HOST"] = required("DS4_TEST_14_WINDOWS_HOST")
    os.environ["DS4_TEST_13_WINDOWS_PORT"] = os.environ.get("DS4_TEST_14_WINDOWS_PORT", "38114")
    if not Path(checkpoint).is_file():
        raise SystemExit(f"Checkpoint TEST_14 non trovato: {checkpoint}")
    test9.TEST_DIR = TEST_DIR
    test9.start_service = start_build7_service
    test9.run_c_with_automatic_audit = run_c_build7
    print("TEST_14: A/B/C del record TEST_13/output_9 con predictor Build 7 Top-4...")
    test9.benchmark_with_automatic_audits(args)


if __name__ == "__main__":
    main()
