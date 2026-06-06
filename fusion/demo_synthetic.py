#!/usr/bin/env python3
"""Dependency-free demo + self-test of the fusion logic (numpy only).

Runs WITHOUT torch / sahi / opencv: it bypasses inference and feeds synthetic
Detection lists straight into the decision logic and Kalman tracker. Use it to
sanity-check the maths of plan §3 (fusion) and §5 (tracking) on any machine,
before wiring up real models with ``run_fusion.py``.

    python fusion/demo_synthetic.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import (  # noqa: E402
    DEFAULT_PARAMS,
    Detection,
    FusionPipeline,
    PipelineConfig,
    Regime,
    associate,
    decision_logic,
    iou,
    noisy_or,
    wbf,
)


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def unit_checks() -> None:
    """Assert the core math matches the plan's formulas."""
    # --- IoU (plan §3.1) ---
    assert _approx(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
    assert _approx(iou([0, 0, 10, 10], [10, 10, 20, 20]), 0.0)
    assert _approx(iou([0, 0, 2, 2], [1, 1, 3, 3]), 1.0 / 7.0)

    # --- noisy-OR (plan §3.3) ---
    assert _approx(noisy_or(0.5, 0.5), 0.75)
    assert _approx(noisy_or(0.0), 0.0)
    assert _approx(noisy_or(0.2, 0.3, 0.4), 1 - 0.8 * 0.7 * 0.6)  # generalised (§8)

    # --- WBF (plan §3.2): DAY weights 0.7 / 0.3, equal conf ---
    p_day = DEFAULT_PARAMS[Regime.DAY]
    e = Detection("EO", [0, 0, 10, 10], 1.0)
    r = Detection("IR", [2, 2, 12, 12], 1.0)
    box = wbf(e, r, p_day)
    assert np.allclose(box, [0.6, 0.6, 10.6, 10.6]), box

    # --- association: one agreeing pair, no leftovers ---
    pairs, eo_only, ir_only = associate([e], [r], p_day.assoc_iou)
    assert len(pairs) == 1 and not eo_only and not ir_only

    # --- agreement vs lonely (plan §3.3) ---
    # Agreement at DAY: conf rises via noisy-OR + bonus, both sensors support it.
    fused = decision_logic([Detection("EO", [0, 0, 10, 10], 0.9)],
                           [Detection("IR", [1, 1, 11, 11], 0.8)],
                           p_day, Regime.DAY)
    assert len(fused) == 1 and fused[0].support == ["EO", "IR"]
    expected = min(1.0, noisy_or(0.7 * 0.9, 0.3 * 0.8) * (1 + p_day.agreement_bonus))
    assert _approx(fused[0].conf, expected), (fused[0].conf, expected)

    # EO-only at NIGHT is suppressed below threshold; IR-only survives.
    p_night = DEFAULT_PARAMS[Regime.NIGHT]
    eo_lonely = decision_logic([Detection("EO", [0, 0, 10, 10], 0.9)], [], p_night, Regime.NIGHT)
    ir_lonely = decision_logic([], [Detection("IR", [0, 0, 10, 10], 0.9)], p_night, Regime.NIGHT)
    assert eo_lonely == [], "EO-only at NIGHT should be dropped"
    assert len(ir_lonely) == 1 and ir_lonely[0].support == ["IR"]

    print("unit_checks: all assertions passed ✓")


def tracking_scenario() -> None:
    """Simulate a moving drone across day->twilight->night and track it."""
    rng = np.random.default_rng(0)

    cfg = PipelineConfig(track_min_hits=3, track_max_age=5)
    pipe = FusionPipeline(cfg)

    # Ground-truth target: starts at (100, 100), 20x20 box, moves (+6, +3)/frame.
    x0, y0, vx, vy, size = 100.0, 100.0, 6.0, 3.0, 20.0

    # Regime schedule by frame (simulated EO mean brightness).
    def mean_v_for(frame):
        if frame < 6:
            return 200.0   # DAY
        if frame < 10:
            return 120.0   # TWILIGHT
        return 60.0        # NIGHT

    print("\nframe regime    meanV  EO IR  fused(support:conf)        track(id@center,vel)")
    print("-" * 92)

    n_frames = 14
    last_track_center = None
    confirmed_seen = False

    for f in range(n_frames):
        cx = x0 + vx * f
        cy = y0 + vy * f
        true_box = [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2]

        # EO detection: reliable by day, drops out at night (misses some frames).
        eo_dets = []
        regime_v = mean_v_for(f)
        eo_present = regime_v > 80 or rng.random() > 0.6
        if eo_present:
            jitter = rng.normal(0, 1.5, 4)
            eo_dets.append(Detection("EO", np.array(true_box) + jitter,
                                     conf=float(np.clip(rng.normal(0.85, 0.05), 0, 1))))

        # IR detection: present throughout, slight co-boresight offset.
        ir_dets = []
        if rng.random() > 0.05:
            offset = np.array([1.5, -1.0, 1.5, -1.0])
            jitter = rng.normal(0, 1.5, 4)
            ir_dets.append(Detection("IR", np.array(true_box) + offset + jitter,
                                     conf=float(np.clip(rng.normal(0.80, 0.05), 0, 1))))

        result = pipe.process_detections(eo_dets, ir_dets, regime_v, frame_ts=float(f))

        fused_str = ", ".join(f"{'+'.join(d.support)}:{d.conf:.2f}" for d in result.fused) or "-"
        if result.tracks:
            t = result.tracks[0]
            c = t.center
            v = t.velocity
            track_str = f"#{t.id}@({c[0]:.0f},{c[1]:.0f}) v=({v[0]:.1f},{v[1]:.1f})"
            last_track_center = c
            confirmed_seen = True
        else:
            track_str = "-"

        print(f"{f:5d} {result.regime.value:8s} {regime_v:5.0f}  "
              f"{len(result.eo_dets):2d} {len(result.ir_dets):2d}  "
              f"{fused_str:26s} {track_str}")

    # --- assertions: a confirmed track formed and is following the target ---
    assert confirmed_seen, "no confirmed track ever formed"
    final_cx = x0 + vx * (n_frames - 1)
    final_cy = y0 + vy * (n_frames - 1)
    assert last_track_center is not None
    err = np.hypot(last_track_center[0] - final_cx, last_track_center[1] - final_cy)
    assert err < 15.0, f"track drifted from target by {err:.1f}px"
    print(f"\ntracking_scenario: confirmed track followed target (final error {err:.1f}px) ✓")


def main() -> int:
    unit_checks()
    tracking_scenario()
    print("\nAll synthetic checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
