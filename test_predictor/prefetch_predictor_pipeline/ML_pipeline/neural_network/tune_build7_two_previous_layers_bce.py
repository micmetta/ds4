#!/usr/bin/env python3
"""Tune Build 7 with the same controlled BCE search used for Build 6.

Build 7 changes only the fixed-NNZ state: it contains the causally available
state of layers l, l-1 and l-2.  Hidden and KV branches remain l and l-1, so
the inherited search space is a controlled comparison with Build 6.
"""

from __future__ import annotations

import sys

import tune_build6_fixed_nnz_previous_layer_ffn as base_tuner


def configure_build7_search_space() -> None:
    """Apply the Build 7-only search space before sampling candidates."""
    # Build 7 has a substantially wider state (13,157 possible features), so
    # the controlled search adds one high-capacity state branch.  The Build 6
    # module retains its original (512, 1024, 2048) search space.
    base_tuner.STATE_HIDDEN_UNITS = (512, 1024, 2048, 4096)


def main() -> None:
    configure_build7_search_space()
    # The shared implementation has no transition-probability input in this
    # experiment.  Appending fixed flags also prevents an accidental switch to
    # the combined ranking objective: Build 7 starts from the classic BCE.
    sys.argv.extend((
        "--ranking-loss-weight", "0.0",
        "--experiment-name", "build7_two_previous_layers_bce",
        "--dataset-label", "Build 7",
    ))
    base_tuner.main()


if __name__ == "__main__":
    main()
