"""Trajectory analysis (plan §5).

A constant-velocity Kalman filter per target with state ``[cx, cy, w, h, vx, vy]``,
plus a SORT-style manager that does gating, confirm-after-N-hits, and
delete-after-K-misses. This is what turns per-frame fused detections into stable
tracks and suppresses one-frame false positives.

Implemented in pure numpy so it has no filterpy/scipy dependency. Association is
greedy IoU matching (a Hungarian assignment could be dropped in if scipy is
available, but greedy is plenty for the small number of targets here).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .fusion import iou
from .schema import FusedDetection

__all__ = ["KalmanBoxTracker", "Track", "Tracker"]


def _state_to_xyxy(x: np.ndarray) -> np.ndarray:
    """state [cx,cy,w,h,...] -> [x1,y1,x2,y2]."""
    cx, cy, w, h = x[0], x[1], max(1.0, x[2]), max(1.0, x[3])
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter for one bounding box (plan §5).

    State x = [cx, cy, w, h, vx, vy]^T ; measurement z = [cx, cy, w, h]^T.
    """

    def __init__(self, bbox, dt: float = 1.0):
        self.dt = float(dt)

        # State transition (constant velocity).
        self.F = np.eye(6)
        self.F[0, 4] = self.dt
        self.F[1, 5] = self.dt

        # Measurement matrix (observe cx, cy, w, h).
        self.H = np.zeros((4, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Covariances. High initial uncertainty on the unobserved velocities.
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0])   # process noise
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])              # measurement noise

        cx, cy, w, h = self._to_cxcywh(bbox)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _to_cxcywh(bbox):
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h

    def predict(self) -> np.ndarray:
        """Advance the state one step; returns the predicted bbox [x1,y1,x2,y2]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Keep w, h positive.
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return _state_to_xyxy(self.x)

    def update(self, bbox) -> None:
        """Correct the state with a measured bbox."""
        cx, cy, w, h = self._to_cxcywh(bbox)
        z = np.array([cx, cy, w, h], dtype=np.float64)

        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def peek_prediction(self) -> np.ndarray:
        """One-step-ahead bbox WITHOUT mutating state (for feedback priors)."""
        return _state_to_xyxy(self.F @ self.x)

    @property
    def bbox(self) -> np.ndarray:
        return _state_to_xyxy(self.x)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.x[4], self.x[5]], dtype=np.float32)


class Track:
    """A tracked target: a Kalman filter plus bookkeeping (plan §5)."""

    _next_id = 1

    def __init__(self, det: FusedDetection, dt: float = 1.0):
        self.id = Track._next_id
        Track._next_id += 1

        self.kf = KalmanBoxTracker(det.bbox_xyxy, dt=dt)
        self.class_id = det.class_id
        self.conf = det.conf
        self.support = list(det.support)
        self.regime = det.regime

        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = False

    def predict(self) -> np.ndarray:
        bbox = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return bbox

    def update(self, det: FusedDetection) -> None:
        self.kf.update(det.bbox_xyxy)
        self.class_id = det.class_id
        self.conf = det.conf
        self.support = list(det.support)
        self.regime = det.regime
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    @property
    def bbox(self) -> np.ndarray:
        return self.kf.bbox

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.kf.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity


class Tracker:
    """Multi-target track manager: gate -> update/seed -> confirm/delete (plan §5)."""

    def __init__(
        self,
        iou_gate: float = 0.3,
        min_hits: int = 3,
        max_age: int = 5,
        dt: float = 1.0,
    ):
        self.iou_gate = float(iou_gate)
        self.min_hits = int(min_hits)
        self.max_age = int(max_age)
        self.dt = float(dt)
        self.tracks: List[Track] = []
        self.frame_count = 0

    def _match(self, predictions: List[np.ndarray], dets: Sequence[FusedDetection]):
        """Greedy IoU matching between predicted track boxes and detections."""
        candidates = []  # (iou, track_idx, det_idx)
        for ti, pred in enumerate(predictions):
            for di, det in enumerate(dets):
                score = iou(pred, det.bbox_xyxy)
                if score >= self.iou_gate:
                    candidates.append((score, ti, di))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_t, matched_d = {}, set()
        for _, ti, di in candidates:
            if ti in matched_t or di in matched_d:
                continue
            matched_t[ti] = di
            matched_d.add(di)

        unmatched_d = [di for di in range(len(dets)) if di not in matched_d]
        return matched_t, unmatched_d

    def update(self, fused_dets: Sequence[FusedDetection]) -> List[Track]:
        """Advance one frame with the current fused detections.

        Returns the confirmed, currently-alive tracks (smoothed bbox, velocity,
        stable id). Newly-seeded tentative tracks are withheld until they reach
        ``min_hits`` so one-frame false positives never surface.
        """
        self.frame_count += 1

        predictions = [t.predict() for t in self.tracks]
        matched_t, unmatched_d = self._match(predictions, fused_dets)

        for ti, di in matched_t.items():
            self.tracks[ti].update(fused_dets[di])

        for di in unmatched_d:
            self.tracks.append(Track(fused_dets[di], dt=self.dt))

        # Confirm / delete.
        for t in self.tracks:
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.confirmed and t.time_since_update <= self.max_age]

    def prior_boxes(self) -> List[np.ndarray]:
        """One-step predictions of confirmed tracks — priors for plan §5 feedback.

        Non-mutating: safe to call before :meth:`update` to tell the decision
        logic where existing targets are expected this frame.
        """
        return [t.kf.peek_prediction() for t in self.tracks if t.confirmed]

    def reset(self) -> None:
        self.tracks = []
        self.frame_count = 0
