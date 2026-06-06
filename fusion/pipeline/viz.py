"""Drawing helpers for the fusion pipeline (debug / demo overlays).

Colour key:
  - EO raw detections   : cyan
  - IR raw detections   : orange
  - fused detections    : magenta
  - confirmed tracks    : green (with id, velocity, support)
Requires OpenCV; imported lazily so the rest of the package stays cv2-free.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .schema import Detection, FusedDetection
from .pipeline import FrameResult

__all__ = ["draw_result"]

_EO_COLOR = (255, 255, 0)      # cyan (BGR)
_IR_COLOR = (0, 165, 255)      # orange
_FUSED_COLOR = (255, 0, 255)   # magenta
_TRACK_COLOR = (0, 255, 0)     # green


def _box(img, bbox, color, thickness=2):
    import cv2

    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return x1, y1


def _label(img, text, org, color):
    import cv2

    x, y = org
    y = max(15, y)
    cv2.putText(img, text, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_result(
    frame: np.ndarray,
    result: FrameResult,
    class_names: Sequence[str],
    show_raw: bool = True,
) -> np.ndarray:
    """Draw detections, fused boxes and confirmed tracks onto a copy of ``frame``."""
    import cv2

    img = frame.copy()

    if show_raw:
        for d in result.eo_dets:
            _box(img, d.bbox_xyxy, _EO_COLOR, 1)
        for d in result.ir_dets:
            _box(img, d.bbox_xyxy, _IR_COLOR, 1)

    for f in result.fused:
        x1, y1 = _box(img, f.bbox_xyxy, _FUSED_COLOR, 2)
        _label(img, f"fuse {f.conf:.2f} {'+'.join(f.support)}", (x1, y1), _FUSED_COLOR)

    for t in result.tracks:
        x1, y1 = _box(img, t.bbox, _TRACK_COLOR, 2)
        cname = class_names[t.class_id] if t.class_id < len(class_names) else str(t.class_id)
        vx, vy = t.velocity
        _label(img, f"#{t.id} {cname} v=({vx:.0f},{vy:.0f})", (x1, y1 + 2), _TRACK_COLOR)

    # Header: regime + brightness + counts.
    mv = "" if np.isnan(result.mean_v) else f" meanV={result.mean_v:.0f}"
    stale = " STALE" if result.stale else ""
    header = (
        f"{result.regime.value}{mv}{stale} | EO:{len(result.eo_dets)} "
        f"IR:{len(result.ir_dets)} fused:{len(result.fused)} tracks:{len(result.tracks)}"
    )
    cv2.putText(img, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img
