#!/usr/bin/env python3
"""Train candidate_10 with a gated residual use of transition probabilities.

This wrapper preserves the historical direct-concatenation training script.  It
only enables the new architecture and forces a fresh initialization, because
the output layers are intentionally incompatible with its phase-20 checkpoint.
"""

from __future__ import annotations

import sys

import train_build6_expert_transition_probabilities_bce as base_training


sys.argv.extend(("--from-scratch", "--expert-transition-gated-residual"))


if __name__ == "__main__":
    base_training.main()
