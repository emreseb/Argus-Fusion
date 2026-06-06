"""Common detection schema (plan §4).

Everything downstream of inference speaks one structure. These are plain
dataclasses (not dicts) so attribute access is checked and the fields are
documented in one place, but the field names match the schema in the plan
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

__all__ = ["Detection", "FusedDetection"]


def _as_xyxy(bbox) -> np.ndarray:
    """Coerce a bbox into a float32 [x1, y1, x2, y2] numpy array."""
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"bbox must have 4 values [x1,y1,x2,y2], got {arr.size}")
    return arr


@dataclass
class Detection:
    """A single per-sensor detection in the shared (co-boresighted) image frame.

    Mirrors the ``Detection`` schema in plan §4.
    """

    source: str                      # "EO" | "IR"
    bbox_xyxy: np.ndarray            # shared-frame pixels [x1, y1, x2, y2]
    conf: float                      # raw model confidence
    class_id: int = 0
    frame_ts: float = 0.0            # timestamp used for sync

    def __post_init__(self) -> None:
        self.bbox_xyxy = _as_xyxy(self.bbox_xyxy)
        self.conf = float(self.conf)
        self.class_id = int(self.class_id)
        self.frame_ts = float(self.frame_ts)

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


@dataclass
class FusedDetection:
    """A fused, single-confidence detection for one frame (plan §4).

    ``class_id`` is not in the plan's minimal schema but is carried through so
    that tracking and visualisation know what was detected.
    """

    bbox_xyxy: np.ndarray            # fused box [x1, y1, x2, y2]
    conf: float                      # fused confidence
    support: List[str]               # which sensors backed it, e.g. ["EO", "IR"]
    regime: str                      # "DAY" | "TWILIGHT" | "NIGHT"
    class_id: int = 0

    def __post_init__(self) -> None:
        self.bbox_xyxy = _as_xyxy(self.bbox_xyxy)
        self.conf = float(self.conf)
        self.class_id = int(self.class_id)

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
