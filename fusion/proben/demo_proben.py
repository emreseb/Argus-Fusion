#!/usr/bin/env python3
"""Dependency-free demo: ProbEn vs the plan's noisy-OR fusion (numpy only).

Shows how the two late-fusion score rules behave on the same inputs, so you can
see *why* you'd pick one over the other before running the full eval.

    python fusion/proben/demo_proben.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import DEFAULT_PARAMS, Detection, Regime, decision_logic  # noqa: E402
from proben.proben import ProbEnConfig, proben_two  # noqa: E402


def show(title, eo, ir):
    p_day = DEFAULT_PARAMS[Regime.DAY]
    pe = proben_two(eo, ir, ProbEnConfig(num_classes=2, assoc_iou=0.4))
    plan = decision_logic(eo, ir, p_day, Regime.DAY)
    pe_s = ", ".join(f"{'+'.join(d.support)}:{d.conf:.3f}" for d in pe) or "-"
    pl_s = ", ".join(f"{'+'.join(d.support)}:{d.conf:.3f}" for d in plan) or "-"
    print(f"\n{title}")
    print(f"  ProbEn (Bayesian product): {pe_s}")
    print(f"  plan   (noisy-OR @ DAY)  : {pl_s}")


def main() -> int:
    print("=" * 64)
    print("ProbEn vs noisy-OR — same detections, different score rules")
    print("=" * 64)

    show("Both sensors agree, conf 0.8 each",
         [Detection("EO", [0, 0, 10, 10], 0.8, 0)],
         [Detection("IR", [1, 1, 11, 11], 0.8, 0)])

    show("Strong + weak agreement (0.9 EO, 0.4 IR)",
         [Detection("EO", [0, 0, 10, 10], 0.9, 0)],
         [Detection("IR", [1, 1, 11, 11], 0.4, 0)])

    show("EO only (0.8) — note ProbEn keeps it, plan penalises lonely",
         [Detection("EO", [0, 0, 10, 10], 0.8, 0)],
         [])

    show("Class disagreement at same spot (EO drone 0.7 / IR bird 0.6)",
         [Detection("EO", [0, 0, 10, 10], 0.7, 0)],
         [Detection("IR", [0, 0, 10, 10], 0.6, 1)])

    print("\nKey differences:")
    print("  - ProbEn product rewards agreement harder (0.8,0.8 -> 0.94).")
    print("  - ProbEn has no regime weighting; the plan biases EO/IR by lighting.")
    print("  - ProbEn collapses confidence on class disagreement (background wins).")
    print("  - The plan discounts lonely single-sensor dets; ProbEn keeps the raw score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
