#!/usr/bin/env python3
"""Build 6 A100 tuner with causal expert-transition probabilities and BCE only.

This is deliberately a thin configuration wrapper around the historical
successive-halving tuner.  The original tuner and all its past experiments
remain reproducible; this file only selects the requested search space and
requires the new 256D transition-probability input.
"""

from __future__ import annotations

import sys

import tune_build6_fixed_nnz_previous_layer_ffn as base_tuner


# Search space agreed for the existing branches.
base_tuner.STATE_HIDDEN_UNITS = (512, 1024, 2048)
base_tuner.HIDDEN_PROJECTION_DIMENSIONS = (64, 512, 1024)
base_tuner.KV_PROJECTION_CHOICES = ("direct", "64", "128", "256")
base_tuner.FUSION_HIDDEN_UNITS = (0, 512, 1024, 2048)

# New causal 256D branch: direct concatenation, or a learned ReLU projection.
base_tuner.EXPERT_TRANSITION_PROJECTION_CHOICES = ("direct", "128", "64")

# This experiment must always use the historical BCE, never the ranking loss.
# Appending makes these values win over accidental command-line duplicates.
sys.argv.extend(("--ranking-loss-weight", "0.0"))


if __name__ == "__main__":
    base_tuner.main()
