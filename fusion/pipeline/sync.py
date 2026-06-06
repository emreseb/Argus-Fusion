"""Frame sync + sizing (plan §2.1).

Temporal sync pairs EO and IR frames by timestamp; **no spatial alignment** is
performed (the sensors are assumed co-boresighted, plan scope note). If the two
streams differ in resolution we resize them to a common size so pixel
coordinates — and therefore IoU in the decision logic — are comparable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["sync_and_size", "nearest_timestamp"]


def sync_and_size(
    eo_img: np.ndarray,
    ir_img: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Put EO and IR into a shared pixel frame (resize only — no warping).

    - If ``target_size`` (w, h) is given, both frames are resized to it.
    - Otherwise, if the frames differ in size, IR is resized onto the EO frame
      so their coordinates line up.
    - If they already match (or target_size is None and sizes agree) they pass
      through untouched, each model getting its native frame.
    """
    import cv2  # lazy

    if target_size is not None:
        eo = cv2.resize(eo_img, target_size)
        ir = cv2.resize(ir_img, target_size)
        return eo, ir

    if eo_img.shape[:2] != ir_img.shape[:2]:
        h, w = eo_img.shape[:2]
        ir_img = cv2.resize(ir_img, (w, h))
    return eo_img, ir_img


def nearest_timestamp(
    target_ts: float,
    candidate_ts,
    tolerance: float,
) -> Optional[int]:
    """Index of the candidate timestamp nearest ``target_ts`` within tolerance.

    Returns ``None`` if the closest candidate is farther than ``tolerance``
    (the stream has stalled — caller should hold the last frame and flag it
    stale rather than fuse mismatched moments, plan §2.1 / §9).
    """
    if len(candidate_ts) == 0:
        return None
    arr = np.asarray(candidate_ts, dtype=np.float64)
    idx = int(np.argmin(np.abs(arr - target_ts)))
    if abs(arr[idx] - target_ts) > tolerance:
        return None
    return idx
