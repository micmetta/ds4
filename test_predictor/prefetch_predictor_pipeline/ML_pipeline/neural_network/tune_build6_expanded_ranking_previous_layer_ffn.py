#!/usr/bin/env python3
"""Expanded Build 6 ranking-loss tuner, isolated from the historical tuner."""

from __future__ import annotations

import sys

import tune_build6_fixed_nnz_previous_layer_ffn as base_tuner


# The original tuner remains untouched.  This experiment only broadens the
# state and post-concatenation choices requested for the A100 run.
base_tuner.STATE_HIDDEN_UNITS = (512, 1024, 2048, 4096)
base_tuner.FUSION_HIDDEN_UNITS = (0, 512, 1024, 2048, 4096)

# Keep the ranking-loss experiment exactly reproducible.  Appending these
# options makes them win over any accidental values passed on the command line.
sys.argv.extend((
    "--ranking-loss-weight", "1.0",
    "--ranking-loss-margin", "0.05",
))


if __name__ == "__main__":
    base_tuner.main()
