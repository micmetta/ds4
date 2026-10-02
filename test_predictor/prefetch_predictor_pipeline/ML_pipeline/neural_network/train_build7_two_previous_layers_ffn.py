#!/usr/bin/env python3
"""Train one selected Build 7 candidate with the classic BCE objective.

The implementation is deliberately shared with the stable Build 6 trainer;
the fixed-NNZ metadata determines the 13,157-dimensional Build 7 state input.
"""

from __future__ import annotations

import sys

import train_build6_fixed_nnz_previous_layer_ffn as base_trainer


def main() -> None:
    # Build 7 is the controlled l-2-state experiment, hence classic BCE only.
    sys.argv.extend((
        "--ranking-loss-weight", "0.0",
        "--experiment-name", "build7_two_previous_layers_bce_training",
        "--dataset-label", "Build 7",
    ))
    base_trainer.main()


if __name__ == "__main__":
    main()
