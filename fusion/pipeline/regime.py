"""Brightness check -> lighting regime (plan §2.2).

Generalises ``brighntess/day_night.py`` from a binary day/night decision into a
three-way DAY / TWILIGHT / NIGHT bucketing with two thresholds, plus optional
EMA smoothing so a passing headlight or cloud doesn't flip the regime every
frame.

The regime is computed on the **EO** stream only: mean(V) of HSV matches
human-visible illumination, whereas IR brightness reflects heat (plan §2.2).

Note on colour order: OpenCV loads BGR. The V (value) channel of HSV is
max(B, G, R), which is identical whether the source is treated as BGR or RGB,
so we convert straight from BGR and skip the BGR->RGB hop that day_night.py
did. Only the H channel would differ, and we never use it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .params import Regime, DEFAULT_T_LOW, DEFAULT_T_HIGH

__all__ = ["mean_brightness_v", "classify_regime", "RegimeEstimator"]


def mean_brightness_v(image_bgr: np.ndarray, resize: Optional[tuple] = (256, 256)) -> float:
    """Return mean HSV value (brightness) of a BGR image in [0, 255].

    ``resize`` downsamples first purely for speed (the mean is scale-stable);
    pass ``None`` to compute on the full frame.
    """
    import cv2  # lazy: keep the package importable without OpenCV

    if image_bgr is None:
        raise ValueError("image_bgr is None")

    img = image_bgr
    if resize is not None:
        img = cv2.resize(img, resize)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def classify_regime(
    mean_v: float,
    t_low: float = DEFAULT_T_LOW,
    t_high: float = DEFAULT_T_HIGH,
) -> Regime:
    """Bucket a mean-brightness value into a regime (plan §2.2 table)."""
    if mean_v >= t_high:
        return Regime.DAY
    if mean_v < t_low:
        return Regime.NIGHT
    return Regime.TWILIGHT


class RegimeEstimator:
    """Stateful EO brightness -> regime estimator with EMA smoothing.

    ``ema_alpha`` is the weight of the newest frame (1.0 == no smoothing).
    """

    def __init__(
        self,
        t_low: float = DEFAULT_T_LOW,
        t_high: float = DEFAULT_T_HIGH,
        ema_alpha: float = 0.3,
        resize: Optional[tuple] = (256, 256),
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.t_low = float(t_low)
        self.t_high = float(t_high)
        self.ema_alpha = float(ema_alpha)
        self.resize = resize
        self._ema: Optional[float] = None
        self.last_mean_v: float = 0.0

    @property
    def smoothed_v(self) -> Optional[float]:
        return self._ema

    def update(self, eo_image_bgr: np.ndarray) -> Regime:
        """Ingest the next EO frame, update the EMA, and return the regime."""
        raw = mean_brightness_v(eo_image_bgr, resize=self.resize)
        self.last_mean_v = raw
        if self._ema is None:
            self._ema = raw
        else:
            self._ema = self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._ema
        return classify_regime(self._ema, self.t_low, self.t_high)

    def reset(self) -> None:
        self._ema = None
        self.last_mean_v = 0.0
