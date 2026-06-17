"""Fuzzy sensor-trust engine — smooth EO/IR weighting (replaces the regime table).

Today the pipeline picks a *single* (modality_weight_EO, modality_weight_IR)
pair by bucketing brightness into DAY/TWILIGHT/NIGHT and reading the §6 table
(see ``params.py``). That is a hard switch: brightness 89 and 91 get different
weights with nothing in between, so trust can snap frame-to-frame at dusk.

This module replaces the *lookup* with a small fuzzy inference system (a Sugeno
one — the rules output plain numbers we use directly as a weight). It answers a
single question — **how much do we trust IR right now?** — from two graded cues:

    1. brightness   dark / dim / bright   (the same EO mean(V) the regime uses)
    2. target size  small / medium / large (long side of the candidate box, px)

and blends the rules smoothly, so the weight *glides* across dusk instead of
jumping. Output is one number ``trust_IR`` in [0, 1]; the EO weight is
``1 - trust_IR``. That maps straight onto the two ``modality_weight_*`` fields
the rest of the pipeline already consumes.

Why these two cues: brightness is the lighting axis (day/night lives here, as
*one* input); target size is a stand-in for "how resolvable is this object" — a
far, tiny drone is a few dark pixels to EO but still a clear hot dot to IR,
while a big, close drone gives EO rich shape to work with. Detection confidence
is deliberately *not* an input here: it already shapes the result inside WBF and
noisy-OR (``fusion.py``), and feeding it in twice would double-count it.

The rule base (the whole brain — nine plain statements):

    trust_IR        size:  small  medium  large
    brightness dark        0.90   0.85    0.80     <- night: lean on thermal
               dim         0.80   0.70    0.60     <- twilight
               bright      0.60   0.45    0.35     <- day: EO earns trust

Read it and it just makes sense: darker -> trust IR more (top to bottom),
smaller/farther -> trust IR more (right to left).

Typical use, behind the existing regime estimator::

    est = RegimeEstimator(...)
    regime = est.update(eo_img)              # still gives the regime label
    ft = FuzzyTrust()
    w_eo, w_ir = ft.weights(est.smoothed_v,  # reuse the EMA-smoothed brightness
                            size_px=box_long_side(det.bbox_xyxy))

Pure standard library (the inference is min/weighted-average); no numpy needed,
so the logic is trivially testable on its own — run this file directly for a
self-test + demo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

__all__ = [
    "FuzzyTrustConfig",
    "FuzzyTrust",
    "box_long_side",
    "load_fuzzy_config",
    "BRIGHT_TERMS",
    "SIZE_TERMS",
]

# Linguistic terms, indexed 0/1/2 to line up with the rule grid's rows/cols.
BRIGHT_TERMS = ("dark", "dim", "bright")
SIZE_TERMS = ("small", "medium", "large")


def _triple_membership(x: float, knots: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Membership of ``x`` in a 3-term partition defined by ascending ``knots``.

    Given ``knots = (a, b, c)`` this returns ``(low, mid, high)`` shaped like::

        low  ‾‾‾\\           mid     /\\          high          /‾‾‾
                 \\                  /  \\                      /
        __________\\____   ____/________\\____    ________/________
                  a   b        a    b    c              b    c

    i.e. ``low`` is a left shoulder fading out by ``b``, ``high`` a right
    shoulder fading in from ``b``, ``mid`` a triangle peaking at ``b``. The
    three always sum to 1, so the blend is a clean weighted average. Adjacent
    terms cross over at the midpoints ``(a+b)/2`` and ``(b+c)/2`` — that is where
    you put the old crisp thresholds to get their smooth equivalent.
    """
    a, b, c = knots
    if x <= a:
        return (1.0, 0.0, 0.0)
    if x < b:
        t = (x - a) / (b - a)
        return (1.0 - t, t, 0.0)
    if x < c:
        t = (x - b) / (c - b)
        return (0.0, 1.0 - t, t)
    return (0.0, 0.0, 1.0)


@dataclass
class FuzzyTrustConfig:
    """Knobs for :class:`FuzzyTrust` — all tunable, none sacred.

    The defaults are sensible starting points: tune ``rules`` and ``size_knots``
    on a *validation* set (not the test set). ``bright_knots`` are chosen so the
    dark/dim and dim/bright crossovers fall on the regime's 90 and 150
    thresholds, i.e. this is the smooth version of the existing buckets.
    """

    # Brightness mean(V) in [0, 255]. Crossovers at (60+120)/2=90 and
    # (120+180)/2=150 — exactly DEFAULT_T_LOW / DEFAULT_T_HIGH.
    bright_knots: Tuple[float, float, float] = (60.0, 120.0, 180.0)
    # Target long-side in pixels (in the working/IR frame). Crossovers at 32px
    # and 72px. Guesses — depends on your typical target range; tune them.
    size_knots: Tuple[float, float, float] = (16.0, 48.0, 96.0)
    # trust_IR consequents, rows = brightness term, cols = size term.
    rules: Tuple[Tuple[float, float, float], ...] = (
        (0.90, 0.85, 0.80),   # dark   : small, medium, large
        (0.80, 0.70, 0.60),   # dim
        (0.60, 0.45, 0.35),   # bright
    )
    # Fallback if (impossibly) no rule fires; 0.5 == trust neither side more.
    default_trust: float = 0.5


class FuzzyTrust:
    """Brightness + target size -> a smooth EO/IR trust weighting.

    Stateless: the smoothing already lives in :class:`RegimeEstimator`, so feed
    this the EMA-smoothed brightness to avoid smoothing twice.
    """

    def __init__(self, cfg: Optional[FuzzyTrustConfig] = None) -> None:
        self.cfg = cfg or FuzzyTrustConfig()

    def memberships(self, brightness: float, size_px: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """Degree to which the inputs belong to each term (handy for debugging)."""
        if size_px is None:
            size_px = self.cfg.size_knots[1]   # unknown size -> neutral "medium"
        b = _triple_membership(float(brightness), self.cfg.bright_knots)
        s = _triple_membership(float(size_px), self.cfg.size_knots)
        return {
            "brightness": dict(zip(BRIGHT_TERMS, b)),
            "size": dict(zip(SIZE_TERMS, s)),
        }

    def trust_ir(self, brightness: float, size_px: Optional[float] = None) -> float:
        """Fused trust in IR, in [0, 1] (0 = believe EO fully, 1 = believe IR).

        Sugeno inference: every rule fires to a degree (the fuzzy AND = min of
        its two memberships); the output is the firing-strength-weighted average
        of the rules' consequents — the "dimmer" that blends the grid.
        """
        if size_px is None:
            size_px = self.cfg.size_knots[1]
        b = _triple_membership(float(brightness), self.cfg.bright_knots)
        s = _triple_membership(float(size_px), self.cfg.size_knots)

        num = 0.0
        den = 0.0
        for i in range(3):            # brightness term
            for j in range(3):        # size term
                strength = b[i] if b[i] < s[j] else s[j]   # fuzzy AND (min)
                if strength > 0.0:
                    num += strength * self.cfg.rules[i][j]
                    den += strength
        if den <= 0.0:
            return self.cfg.default_trust
        return num / den

    def weights(self, brightness: float, size_px: Optional[float] = None) -> Tuple[float, float]:
        """Return ``(modality_weight_EO, modality_weight_IR)`` for this context."""
        t = self.trust_ir(brightness, size_px)
        return (1.0 - t, t)


def box_long_side(bbox_xyxy: Sequence[float]) -> float:
    """Longer side of an ``[x1, y1, x2, y2]`` box — the size cue for the engine."""
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    return max(abs(x2 - x1), abs(y2 - y1))


def load_fuzzy_config(path: str) -> FuzzyTrustConfig:
    """Load a :class:`FuzzyTrustConfig` from JSON, inheriting any omitted field.

    Mirrors ``params.load_params``: a partial file overrides only what it names.
    Example JSON::

        {"size_knots": [12, 40, 80],
         "rules": [[0.92, 0.88, 0.82], [0.82, 0.72, 0.62], [0.62, 0.48, 0.38]]}
    """
    with open(path, "r") as fh:
        raw = json.load(fh)

    base = FuzzyTrustConfig()
    known = {"bright_knots", "size_knots", "rules", "default_trust"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown fuzzy-trust field(s): {sorted(unknown)}")

    def _tup(v):
        return tuple(v) if isinstance(v, list) else v

    return FuzzyTrustConfig(
        bright_knots=_tup(raw.get("bright_knots", base.bright_knots)),
        size_knots=_tup(raw.get("size_knots", base.size_knots)),
        rules=tuple(_tup(r) for r in raw["rules"]) if "rules" in raw else base.rules,
        default_trust=float(raw.get("default_trust", base.default_trust)),
    )


# ---------------------------------------------------------------------------
# Self-test + demo:  python fusion/pipeline/fuzzy_trust.py
# ---------------------------------------------------------------------------
def _demo() -> None:
    ft = FuzzyTrust()
    cfg = ft.cfg

    print("Rule base (trust_IR):")
    print(f"  {'':<10}{'small':>8}{'medium':>8}{'large':>8}")
    for i, bt in enumerate(BRIGHT_TERMS):
        row = "".join(f"{cfg.rules[i][j]:>8.2f}" for j in range(3))
        print(f"  {bt:<10}{row}")

    print("\nMembership crossovers (the soft thresholds):")
    print(f"  brightness dark|dim  @ {(cfg.bright_knots[0]+cfg.bright_knots[1])/2:.0f}"
          f"   dim|bright @ {(cfg.bright_knots[1]+cfg.bright_knots[2])/2:.0f}"
          "   (= the old 90 / 150)")
    print(f"  size       small|med @ {(cfg.size_knots[0]+cfg.size_knots[1])/2:.0f}px"
          f"  med|large @ {(cfg.size_knots[1]+cfg.size_knots[2])/2:.0f}px")

    print("\nThe dimmer: trust_IR vs brightness at a fixed medium target (48px) —")
    print("  watch it glide through the old 90 cutoff instead of snapping:")
    for v in (70, 85, 88, 90, 92, 95, 110, 150):
        print(f"    brightness {v:>3} -> trust_IR {ft.trust_ir(v, 48):.3f}")

    print("\nWorked example from the design (brightness 100, target 30px):")
    m = ft.memberships(100, 30)
    print(f"    brightness 100 -> {m['brightness']}")
    print(f"    size       30  -> {m['size']}")
    print(f"    => trust_IR = {ft.trust_ir(100, 30):.3f}"
          f"  (w_EO={ft.weights(100,30)[0]:.3f}, w_IR={ft.weights(100,30)[1]:.3f})")

    # --- assertions: the logic must hold ---
    # Corners collapse onto the grid (single rule fires at strength 1).
    assert abs(ft.trust_ir(0, 0) - 0.90) < 1e-9       # night + small
    assert abs(ft.trust_ir(255, 200) - 0.35) < 1e-9   # day + large
    assert abs(ft.trust_ir(120, 48) - 0.70) < 1e-9    # dead-centre = dim+medium
    # The worked example lands on ~0.80 as designed.
    assert abs(ft.trust_ir(100, 30) - 0.8037) < 1e-3
    # Monotonic: darker => more IR trust; smaller => more IR trust.
    assert ft.trust_ir(40, 48) > ft.trust_ir(200, 48)
    assert ft.trust_ir(120, 10) > ft.trust_ir(120, 200)
    # Smoothness: a 1-unit brightness step never causes a big jump.
    assert abs(ft.trust_ir(90, 48) - ft.trust_ir(91, 48)) < 0.01
    # Weights complement to 1.
    we, wi = ft.weights(100, 30)
    assert abs(we + wi - 1.0) < 1e-9
    print("\nself-test: OK")


if __name__ == "__main__":
    _demo()
