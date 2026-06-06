"""Per-sensor inference wrapper (plan §2.3, §10.1).

Wraps the SAHI sliced-inference pattern from ``live/picture_inference_SAHI.py``
into a reusable ``Detector.run(img, conf) -> [Detection]`` used identically for
the EO and IR weights. Falls back to plain Ultralytics YOLO when slicing is
disabled (faster when targets are large relative to the frame).

All heavy imports (sahi / ultralytics / torch) are deferred to first use so the
rest of the package — schema, fusion math, tracking — imports with numpy alone.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .schema import Detection

__all__ = ["Detector", "resolve_device"]


def resolve_device(device: str = "auto") -> str:
    """Pick a torch device string, honouring an explicit choice."""
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Detector:
    """One YOLO model behind a uniform ``run`` API, labelled EO or IR.

    Parameters mirror the SAHI setup in the existing repo: 640x640 slices with
    20% overlap so small drones aren't cut in half. ``source`` is stamped onto
    every emitted :class:`Detection` ("EO" or "IR").
    """

    def __init__(
        self,
        model_path: str,
        source: str,
        device: str = "auto",
        use_sahi: bool = True,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        base_confidence: float = 0.25,
    ):
        if source not in ("EO", "IR"):
            raise ValueError("source must be 'EO' or 'IR'")
        self.model_path = model_path
        self.source = source
        self.device = resolve_device(device)
        self.use_sahi = use_sahi
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.base_confidence = base_confidence

        self._sahi_model = None   # lazy
        self._yolo = None         # lazy

    # -- lazy model loaders ---------------------------------------------------
    def _load_sahi(self):
        if self._sahi_model is None:
            from sahi import AutoDetectionModel

            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.model_path,
                confidence_threshold=self.base_confidence,
                device=self.device,
            )
        return self._sahi_model

    def _load_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO

            self._yolo = YOLO(self.model_path)
        return self._yolo

    # -- inference ------------------------------------------------------------
    def run(self, image_bgr: np.ndarray, conf: float, frame_ts: float = 0.0) -> List[Detection]:
        """Detect on a BGR frame at confidence ``conf``; return Detections.

        ``conf`` is the regime's per-sensor threshold (plan §6); detections
        below it are dropped here so the decision logic only ever sees the kept
        boxes.
        """
        import cv2

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.use_sahi:
            return self._run_sahi(image_rgb, conf, frame_ts)
        return self._run_yolo(image_rgb, conf, frame_ts)

    def _run_sahi(self, image_rgb, conf, frame_ts) -> List[Detection]:
        from sahi.predict import get_sliced_prediction

        model = self._load_sahi()
        model.confidence_threshold = conf  # adjust per regime

        result = get_sliced_prediction(
            image_rgb,
            model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=False,
        )

        dets: List[Detection] = []
        for op in result.object_prediction_list:
            score = float(op.score.value)
            if score < conf:           # belt-and-braces filter
                continue
            b = op.bbox
            dets.append(
                Detection(
                    source=self.source,
                    bbox_xyxy=[b.minx, b.miny, b.maxx, b.maxy],
                    conf=score,
                    class_id=int(op.category.id),
                    frame_ts=frame_ts,
                )
            )
        return dets

    def _run_yolo(self, image_rgb, conf, frame_ts) -> List[Detection]:
        model = self._load_yolo()
        results = model(image_rgb, conf=conf, device=self.device, verbose=False)

        dets: List[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                dets.append(
                    Detection(
                        source=self.source,
                        bbox_xyxy=[x1, y1, x2, y2],
                        conf=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                        frame_ts=frame_ts,
                    )
                )
        return dets
