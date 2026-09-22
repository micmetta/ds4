#!/usr/bin/env python3
"""TEST_13: A/B/C TEST_9 con il worker SSD sostituito dal bridge Windows."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "test_predictor" / "test_9"))
import test_9_runner as test9  # noqa: E402


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Variabile obbligatoria mancante: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="TEST_13 bridge Windows -> slot CUDA pinned DS4.")
    parser.add_argument("--repetitions", type=int, default=int(os.environ.get("DS4_TEST_13_REPETITIONS", "1")))
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions deve essere positivo.")
    # TEST_9 e' riusato solo come orchestratore A/B/C; TEST_13 conserva la
    # propria directory di output e il proprio namespace delle variabili.
    mapping = {
        "DS4_TEST_9_PREFETCH_MODEL": required("DS4_TEST_13_PREFETCH_MODEL"),
        "DS4_TEST_9_CHECKPOINT": required("DS4_TEST_13_CHECKPOINT"),
        "DS4_TEST_9_PREFETCH_LAYERS": os.environ.get("DS4_TEST_13_PREFETCH_LAYERS", "1"),
        "DS4_TEST_9_PREFETCH_SLOTS_PER_LAYER": os.environ.get("DS4_TEST_13_PREFETCH_SLOTS_PER_LAYER", "1"),
        "DS4_TEST_9_PREFETCH_STEP": os.environ.get("DS4_TEST_13_PREFETCH_STEP", "1"),
        "DS4_TEST_9_PREFETCH_NUM_WORKERS": os.environ.get("DS4_TEST_13_PREFETCH_NUM_WORKERS", "1"),
    }
    os.environ.update(mapping)
    os.environ["DS4_TEST_13_WINDOWS_BRIDGE"] = "1"
    test9.TEST_DIR = TEST_DIR
    print("TEST_13: A/B/C con bridge Windows attivo nel solo worker learned prefetch...")
    test9.benchmark_with_automatic_audits(args)


if __name__ == "__main__":
    main()
