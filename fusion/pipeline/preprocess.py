"""EO low-light preprocessing (plan §2.3 / §6 ``eo_preprocess``).

The active regime can ask for the EO frame to be enhanced before inference:
``gamma`` in twilight, ``clahe+gamma`` at night. This only ever touches the EO
stream — IR is left as-is. Applied to a copy; the input frame is not mutated.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["enhance", "gamma_correct", "clahe"]


def gamma_correct(image_bgr: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Brighten an image with a gamma curve (gamma < 1 lifts shadows)."""
    import cv2  # lazy

    inv = 1.0 / max(gamma, 1e-6)
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image_bgr, lut)


def clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L (LAB) channel."""
    import cv2  # lazy

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = op.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance(image_bgr: np.ndarray, mode: Optional[str], gamma: float = 0.6) -> np.ndarray:
    """Apply the named EO enhancement.

    ``mode`` is one of ``None``/``"none"``, ``"gamma"``, ``"clahe"``,
    ``"clahe+gamma"`` (matching the plan §6 ``eo_preprocess`` column).
    """
    if mode is None or mode == "none":
        return image_bgr

    if mode == "gamma":
        return gamma_correct(image_bgr, gamma)
    if mode == "clahe":
        return clahe(image_bgr)
    if mode == "clahe+gamma":
        return gamma_correct(clahe(image_bgr), gamma)

    raise ValueError(f"Unknown eo_preprocess mode: {mode!r}")
