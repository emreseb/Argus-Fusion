"""Pipeline configuration (one place for every knob).

Holds model paths, inference/slicing settings, brightness thresholds, sync
sizing, tracking gates and the optional trajectory-feedback switch. Construct it
directly or load from JSON with :meth:`PipelineConfig.from_json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

__all__ = ["PipelineConfig"]


@dataclass
class PipelineConfig:
    # --- models (plan §2.3) ---
    eo_model_path: str = "best.pt"
    ir_model_path: str = "best.pt"
    device: str = "auto"                 # "auto" | "cpu" | "cuda:0" | "mps"
    class_names: List[str] = field(default_factory=lambda: ["drone", "bird"])

    # --- inference / SAHI slicing (plan §2.3, picture_inference_SAHI.py) ---
    use_sahi: bool = True
    slice_height: int = 640
    slice_width: int = 640
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2

    # --- brightness regime (plan §2.2, §6) ---
    t_low: float = 90.0
    t_high: float = 150.0
    ema_alpha: float = 0.3
    brightness_resize: Optional[Tuple[int, int]] = (256, 256)

    # --- sync / sizing (plan §2.1) ---
    target_size: Optional[Tuple[int, int]] = None   # (w, h) to resize both, or None

    # --- association (plan §3.1) ---
    require_same_class: bool = False

    # --- box fusion (plan §3.2) ---
    # "wbf": trust/confidence-weighted average of the EO+IR corners (plan default).
    # "select": keep the higher-trust sensor's box verbatim, use the other only
    # for confidence — avoids a low-trust box dragging a high-trust one off-target.
    # The winner follows the same weights, so a stronger EO model shifts it back.
    box_mode: str = "wbf"

    # --- tracking (plan §5) ---
    track_iou_gate: float = 0.3
    track_min_hits: int = 3
    track_max_age: int = 5
    track_dt: float = 1.0

    # --- optional trajectory feedback (plan §5) ---
    use_track_feedback: bool = False
    prior_iou_gate: float = 0.3
    prior_bonus: float = 0.15
    no_prior_penalty: float = 0.20

    # --- optional self-calibrating EO->IR registration (registration.py) ---
    # The plan assumes co-boresighted sensors; enable this when they are not.
    use_registration: bool = False
    registration_matrix_path: Optional[str] = None   # load a pre-fit affine and skip warm-up
    reg_ir_size: Tuple[int, int] = (1280, 1024)       # (w, h) shared coordinate frame = IR
    reg_min_pairs: int = 40                            # correspondences before attempting a fit
    reg_max_residual: float = 8.0                      # lock only below this median error (px)

    # --- optional fuzzy sensor-trust engine (fuzzy_trust.py) ---
    # Replaces the per-regime modality_weight_* lookup with a smooth, size-aware
    # EO/IR trust dial (brightness + target size). Penalties/threshold unchanged.
    use_fuzzy_trust: bool = False
    fuzzy_config_path: Optional[str] = None   # JSON overrides for the rule grid / knots

    # --- optional per-regime param override file (plan §6) ---
    params_path: Optional[str] = None

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path, "r") as fh:
            raw = json.load(fh)
        # tuples survive JSON as lists; coerce the size fields back.
        for key in ("brightness_resize", "target_size", "reg_ir_size"):
            if isinstance(raw.get(key), list):
                raw[key] = tuple(raw[key])
        unknown = set(raw) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"Unknown config key(s): {sorted(unknown)}")
        return cls(**raw)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
