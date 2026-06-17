"""End-to-end fusion pipeline orchestrator (plan §2 diagram, §7 pseudocode).

Ties the stages together:

    sync -> EO brightness regime -> per-model inference (EO/IR) ->
    decision logic (associate/WBF/noisy-OR) -> trajectory (Kalman) tracks.

Two entry points:

- :meth:`FusionPipeline.process_frame` — the real path: give it raw EO and IR
  frames, it runs the detectors and returns the fused detections + confirmed
  tracks for that frame.
- :meth:`FusionPipeline.process_detections` — the model-free path: feed it
  detection lists you already have (used by the synthetic demo and tests, and
  handy for replaying logged detections). Requires only numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np

from .config import PipelineConfig
from .fusion import decision_logic
from .fuzzy_trust import FuzzyTrust, load_fuzzy_config
from .params import DEFAULT_PARAMS, Regime, load_params
from .regime import RegimeEstimator, classify_regime
from .schema import Detection, FusedDetection
from .tracking import Track, Tracker

__all__ = ["FusionPipeline", "FrameResult"]


@dataclass
class FrameResult:
    """Everything produced for a single frame — for output, viz and debugging."""

    regime: Regime
    mean_v: float
    eo_dets: List[Detection]
    ir_dets: List[Detection]
    fused: List[FusedDetection]
    tracks: List[Track]
    frame_ts: float = 0.0
    smoothed_v: float = float("nan")  # EMA-smoothed brightness fed to fuzzy trust
    stale: bool = False               # set when a stream was held (plan §2.1/§9)
    extra: dict = field(default_factory=dict)


class FusionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.params = (
            load_params(self.cfg.params_path) if self.cfg.params_path else dict(DEFAULT_PARAMS)
        )
        self.regime_estimator = RegimeEstimator(
            t_low=self.cfg.t_low,
            t_high=self.cfg.t_high,
            ema_alpha=self.cfg.ema_alpha,
            resize=self.cfg.brightness_resize,
        )
        self.tracker = Tracker(
            iou_gate=self.cfg.track_iou_gate,
            min_hits=self.cfg.track_min_hits,
            max_age=self.cfg.track_max_age,
            dt=self.cfg.track_dt,
        )
        self._eo_detector = None   # lazy (avoids loading torch until needed)
        self._ir_detector = None

        # Optional self-calibrating EO->IR registration (registration.py).
        self.registrar = None
        if self.cfg.use_registration:
            from .registration import AutoRegistrar, RegistrationConfig

            self.registrar = AutoRegistrar(RegistrationConfig(
                ir_size=tuple(self.cfg.reg_ir_size),
                min_pairs=self.cfg.reg_min_pairs,
                max_residual=self.cfg.reg_max_residual,
            ))
            if self.cfg.registration_matrix_path:
                self.registrar.load(self.cfg.registration_matrix_path)

        # Optional fuzzy sensor-trust engine (fuzzy_trust.py): smooth, size-aware
        # EO/IR weighting from brightness, replacing the per-regime weight table.
        self.fuzzy = None
        if self.cfg.use_fuzzy_trust:
            fz_cfg = (
                load_fuzzy_config(self.cfg.fuzzy_config_path)
                if self.cfg.fuzzy_config_path else None
            )
            self.fuzzy = FuzzyTrust(fz_cfg)

    # -- detectors (lazy) -----------------------------------------------------
    def _make_detector(self, model_path: str, source: str):
        from .inference import Detector

        return Detector(
            model_path=model_path,
            source=source,
            device=self.cfg.device,
            use_sahi=self.cfg.use_sahi,
            slice_height=self.cfg.slice_height,
            slice_width=self.cfg.slice_width,
            overlap_height_ratio=self.cfg.overlap_height_ratio,
            overlap_width_ratio=self.cfg.overlap_width_ratio,
        )

    @property
    def eo_detector(self):
        if self._eo_detector is None:
            self._eo_detector = self._make_detector(self.cfg.eo_model_path, "EO")
        return self._eo_detector

    @property
    def ir_detector(self):
        if self._ir_detector is None:
            self._ir_detector = self._make_detector(self.cfg.ir_model_path, "IR")
        return self._ir_detector

    # -- shared fuse + track step --------------------------------------------
    def _fuse_and_track(
        self,
        eo_dets: Sequence[Detection],
        ir_dets: Sequence[Detection],
        regime: Regime,
        mean_v: float,
        frame_ts: float,
        stale: bool = False,
    ) -> FrameResult:
        p = self.params[regime]

        priors = self.tracker.prior_boxes() if self.cfg.use_track_feedback else None

        # Brightness for the fuzzy engine: prefer the EMA-smoothed value; fall
        # back to the raw mean if the estimator was bypassed (process_detections).
        brightness = self.regime_estimator.smoothed_v
        if brightness is None and mean_v == mean_v:   # mean_v == mean_v: not NaN
            brightness = mean_v

        fused = decision_logic(
            eo_dets,
            ir_dets,
            p,
            regime,
            require_same_class=self.cfg.require_same_class,
            priors=priors,
            prior_iou_gate=self.cfg.prior_iou_gate,
            prior_bonus=self.cfg.prior_bonus,
            no_prior_penalty=self.cfg.no_prior_penalty,
            fuzzy=self.fuzzy,
            brightness=brightness,
            box_mode=self.cfg.box_mode,
        )

        tracks = self.tracker.update(fused)

        return FrameResult(
            regime=regime,
            mean_v=mean_v,
            eo_dets=list(eo_dets),
            ir_dets=list(ir_dets),
            fused=fused,
            tracks=tracks,
            frame_ts=frame_ts,
            smoothed_v=brightness if brightness is not None else float("nan"),
            stale=stale,
        )

    # -- real path: run the models -------------------------------------------
    def process_frame(
        self,
        eo_img: np.ndarray,
        ir_img: np.ndarray,
        frame_ts: float = 0.0,
        stale: bool = False,
    ) -> FrameResult:
        """Full per-frame pipeline (plan §7 ``process_frame``).

        With ``use_registration`` the EO and IR models run on their **native**
        frames and EO detections are mapped into the IR coordinate frame by the
        self-calibrating affine (see registration.py); otherwise the frames are
        sync-resized to a shared size as before (co-boresighted assumption).
        """
        from .preprocess import enhance

        if self.registrar is None:
            from .sync import sync_and_size

            eo_img, ir_img = sync_and_size(eo_img, ir_img, target_size=self.cfg.target_size)

        regime = self.regime_estimator.update(eo_img)   # EO brightness (§2.2)
        mean_v = self.regime_estimator.last_mean_v
        p = self.params[regime]

        eo_in = enhance(eo_img, p.eo_preprocess) if p.eo_preprocess else eo_img  # EO only

        eo_dets = self.eo_detector.run(eo_in, conf=p.conf_thresh_EO, frame_ts=frame_ts)
        ir_dets = self.ir_detector.run(ir_img, conf=p.conf_thresh_IR, frame_ts=frame_ts)

        if self.registrar is not None:
            self.registrar.observe(eo_dets, ir_dets, eo_img.shape)   # learn (no-op once locked)
            eo_dets = [
                Detection(d.source, self.registrar.apply(d.bbox_xyxy, eo_img.shape),
                          d.conf, d.class_id, d.frame_ts)
                for d in eo_dets
            ]

        result = self._fuse_and_track(eo_dets, ir_dets, regime, mean_v, frame_ts, stale)
        if self.registrar is not None:
            result.extra["registration"] = {
                "locked": self.registrar.locked,
                "n_pairs": self.registrar.n_pairs,
                "residual_px": self.registrar.last_residual,
            }
        return result

    # -- model-free path: detections already in hand -------------------------
    def process_detections(
        self,
        eo_dets: Sequence[Detection],
        ir_dets: Sequence[Detection],
        regime: Union[Regime, float],
        frame_ts: float = 0.0,
    ) -> FrameResult:
        """Fuse + track pre-computed detections (no models, numpy only).

        ``regime`` may be a :class:`Regime`, a regime name, or a raw mean-V
        float (which is classified with the configured thresholds).
        """
        if isinstance(regime, (int, float)) and not isinstance(regime, bool):
            mean_v = float(regime)
            regime = classify_regime(mean_v, self.cfg.t_low, self.cfg.t_high)
        else:
            if not isinstance(regime, Regime):
                regime = Regime(str(regime))
            mean_v = float("nan")

        return self._fuse_and_track(eo_dets, ir_dets, regime, mean_v, frame_ts)

    def reset(self) -> None:
        """Clear tracking + brightness state (e.g. between clips)."""
        self.tracker.reset()
        self.regime_estimator.reset()
