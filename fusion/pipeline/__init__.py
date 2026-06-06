"""EO/IR detection fusion pipeline.

Late (detection-level), brightness-aware fusion of per-sensor detectors feeding
a Kalman trajectory tracker. Implements docs/fusion-pipeline-plan.md.

Light modules (schema, params, regime, preprocess, sync, fusion, tracking,
pipeline, config) import with numpy only. The heavy bits — the SAHI/Ultralytics
``Detector`` and the OpenCV ``draw_result`` overlay — import their dependencies
lazily, so ``from pipeline import FusionPipeline`` works even before torch is
installed; you only need the ML stack when you actually call ``process_frame``.
"""

from __future__ import annotations

from .config import PipelineConfig
from .fusion import associate, decision_logic, iou, noisy_or, wbf
from .params import (
    DEFAULT_PARAMS,
    DEFAULT_T_HIGH,
    DEFAULT_T_LOW,
    Regime,
    RegimeParams,
    load_params,
)
from .pipeline import FrameResult, FusionPipeline
from .regime import RegimeEstimator, classify_regime, mean_brightness_v
from .schema import Detection, FusedDetection
from .tracking import KalmanBoxTracker, Track, Tracker

__all__ = [
    "PipelineConfig",
    "FusionPipeline",
    "FrameResult",
    "Detection",
    "FusedDetection",
    "Regime",
    "RegimeParams",
    "DEFAULT_PARAMS",
    "DEFAULT_T_LOW",
    "DEFAULT_T_HIGH",
    "load_params",
    "RegimeEstimator",
    "classify_regime",
    "mean_brightness_v",
    "iou",
    "associate",
    "wbf",
    "noisy_or",
    "decision_logic",
    "KalmanBoxTracker",
    "Track",
    "Tracker",
]
