#!/usr/bin/env python3
"""Build 7 Top-4 tuning B1-B3: wider symmetric state/raw-hidden branches.

This is intentionally a separate search from A1-A3.  It evaluates only
train/validation and stops after B3; no test evaluation or automatic
1,000-epoch training is performed.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import tune_build7_top4_successive_halving as base


STATE_UNITS = (4096, 8192)
SYMMETRIC_HIDDEN_UNITS = (1024, 2048, 4096)
FUSION_UNITS = (2048, 4096)
PHASES = (
    ("B1", 5, 12, 6, None),
    ("B2", 20, 6, 3, None),
    ("B3", 60, 3, None, 5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuning Build 7 Top-4 B1-B3: 12 -> 6 -> 3, senza test e senza fase finale automatica."
    )
    parser.add_argument("--state-data-dir", type=Path, required=True)
    parser.add_argument("--current-fixed-nnz-state-data-dir", type=Path, required=True)
    parser.add_argument("--current-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--previous-quantized-hidden-data-dir", type=Path, required=True)
    parser.add_argument("--expert-transition-probabilities-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Riprende dalla prima fase incompleta.")
    parser.add_argument("--l2-regularization", type=float, default=3e-6)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout fisso dopo state e fusione; i raw hidden restano senza dropout.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--shuffle-block-size", type=int, default=65536)
    return parser.parse_args()


def all_candidates() -> list[dict[str, Any]]:
    candidates = []
    for number, values in enumerate(
        itertools.product(STATE_UNITS, SYMMETRIC_HIDDEN_UNITS, FUSION_UNITS), start=1
    ):
        state_units, hidden_units, fusion_units = values
        candidates.append({
            "candidate_id": f"candidate_B{number:02d}",
            "hyperparameters": {
                "state_hidden_units": state_units,
                "current_hidden_projection_dimension": hidden_units,
                "previous_hidden_projection_dimension": hidden_units,
                "fusion_hidden_units": fusion_units,
                "raw_hidden_branch_dropout": 0.0,
            },
            "phases": {},
        })
    if len(candidates) != 12:
        raise AssertionError("Lo spazio Top-4 B deve contenere esattamente 12 candidate.")
    return candidates


def main() -> None:
    # Reuse the hardened trainer/checkpoint implementation from A, changing
    # only the explicit B search space and phase schedule.
    base.STATE_UNITS = STATE_UNITS
    base.CURRENT_HIDDEN_UNITS = SYMMETRIC_HIDDEN_UNITS
    base.PREVIOUS_HIDDEN_UNITS = SYMMETRIC_HIDDEN_UNITS
    base.FUSION_UNITS = FUSION_UNITS
    base.PHASES = PHASES
    base.EXPERIMENT_NAME = "build7_top4_symmetric_capacity_successive_halving_b1_b3"
    base.SEARCH_LABEL = "B"
    base.parse_args = parse_args
    base.all_candidates = all_candidates
    base.main()


if __name__ == "__main__":
    main()
