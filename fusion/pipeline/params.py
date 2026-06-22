"""Per-regime parameter table (plan §6) and the brightness regime enum.

These are the "parameters that get adjusted" in the pipeline diagram. The
default values are copied verbatim from the table in plan §6; they are starting
points meant to be tuned on real day *and* night footage (plan §10.5).
Override them with :func:`load_params` from a JSON file without touching code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Optional

__all__ = [
    "Regime",
    "RegimeParams",
    "DEFAULT_PARAMS",
    "DEFAULT_T_LOW",
    "DEFAULT_T_HIGH",
    "load_params",
    "params_to_dict",
]


class Regime(str, Enum):
    """Lighting regime selected by the EO brightness check (plan §2.2)."""

    DAY = "DAY"
    TWILIGHT = "TWILIGHT"
    NIGHT = "NIGHT"


# Brightness thresholds on mean(V). day_night.py uses 140; plan §6 suggests
# starting a touch wider so TWILIGHT has room.
DEFAULT_T_LOW = 90.0
DEFAULT_T_HIGH = 150.0


@dataclass
class RegimeParams:
    """The full parameter set active for one lighting regime (plan §6)."""

    modality_weight_EO: float       # trust in EO
    modality_weight_IR: float       # trust in IR
    conf_thresh_EO: float           # EO detection confidence cut
    conf_thresh_IR: float           # IR detection confidence cut
    eo_preprocess: Optional[str]    # None | "gamma" | "clahe+gamma" (EO only)
    assoc_iou: float                # EO/IR association IoU threshold
    agreement_bonus: float          # reward for independent confirmation
    lonely_penalty_EO: float        # discount an EO-only detection
    lonely_penalty_IR: float        # discount an IR-only detection
    decision_threshold: float       # keep fused detection if conf >= this


# ----------------------------------------------------------------------------
# Plan §6 table, transcribed exactly.
#
# | Parameter            | DAY  | TWILIGHT | NIGHT            |
# |----------------------|------|----------|------------------|
# | modality_weight_EO   | 0.70 | 0.50     | 0.30             |
# | modality_weight_IR   | 0.30 | 0.50     | 0.70             |
# | conf_thresh_EO       | 0.40 | 0.45     | 0.55             |
# | conf_thresh_IR       | 0.45 | 0.40     | 0.35             |
# | eo_preprocess        | none | gamma    | clahe+gamma      |
# | assoc_iou            | 0.40 | 0.40     | 0.40             |
# | agreement_bonus      | 0.15 | 0.20     | 0.20             |
# | lonely_penalty       | 0.20 | 0.15     | EO 0.40 / IR 0.10|
# | decision_threshold   | 0.45 | 0.45     | 0.45             |
# ----------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[Regime, RegimeParams] = {
    Regime.DAY: RegimeParams(
        modality_weight_EO=0.70,
        modality_weight_IR=0.30,
        conf_thresh_EO=0.40,
        conf_thresh_IR=0.45,
        eo_preprocess=None,
        assoc_iou=0.40,
        agreement_bonus=0.15,
        lonely_penalty_EO=0.20,
        lonely_penalty_IR=0.20,
        decision_threshold=0.45,
    ),
    Regime.TWILIGHT: RegimeParams(
        modality_weight_EO=0.50,
        modality_weight_IR=0.50,
        conf_thresh_EO=0.45,
        conf_thresh_IR=0.40,
        eo_preprocess="gamma",
        assoc_iou=0.40,
        agreement_bonus=0.20,
        lonely_penalty_EO=0.15,
        lonely_penalty_IR=0.10,   # low light: trust a lone IR detection (toward NIGHT)
        decision_threshold=0.45,
    ),
    Regime.NIGHT: RegimeParams(
        modality_weight_EO=0.30,
        modality_weight_IR=0.70,
        conf_thresh_EO=0.55,
        conf_thresh_IR=0.35,
        eo_preprocess="clahe+gamma",
        assoc_iou=0.40,
        agreement_bonus=0.20,
        lonely_penalty_EO=0.40,   # EO at night should rarely stand alone
        lonely_penalty_IR=0.10,   # IR at night is trusted, light penalty
        decision_threshold=0.45,
    ),
}


def params_to_dict(params: Dict[Regime, RegimeParams]) -> dict:
    """Serialise a regime->params mapping into a plain JSON-friendly dict."""
    return {regime.value: asdict(p) for regime, p in params.items()}


def load_params(path: str) -> Dict[Regime, RegimeParams]:
    """Load a per-regime parameter table from JSON, falling back to defaults.

    The JSON is keyed by regime name ("DAY"/"TWILIGHT"/"NIGHT"); any field left
    out inherits the default for that regime, so partial override files work.
    """
    with open(path, "r") as fh:
        raw = json.load(fh)

    out: Dict[Regime, RegimeParams] = {}
    for regime in Regime:
        base = asdict(DEFAULT_PARAMS[regime])
        override = raw.get(regime.value, {})
        unknown = set(override) - set(base)
        if unknown:
            raise ValueError(f"Unknown param(s) for {regime.value}: {sorted(unknown)}")
        base.update(override)
        out[regime] = RegimeParams(**base)
    return out
