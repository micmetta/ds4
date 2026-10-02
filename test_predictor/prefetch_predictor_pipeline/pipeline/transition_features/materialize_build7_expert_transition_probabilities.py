#!/usr/bin/env python3
"""Materialize the causal transition-probability feature aligned to Build 7.

The underlying estimator is shared with Build 6 because the definition of
P_l(i | j), Laplace smoothing and prompt-level out-of-fold protection are
identical.  This wrapper makes the Build 7 inputs explicit and avoids an
accidental attempt to reuse arrays with a different row order.
"""

from __future__ import annotations

import sys

import materialize_expert_transition_probabilities as base_materializer


def main() -> None:
    base_materializer.main()


if __name__ == "__main__":
    main()
