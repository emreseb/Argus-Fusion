"""Self-calibrating EO->IR spatial registration (auto-affine).

The plan (docs/fusion-pipeline-plan.md §2.1) assumes the EO and IR sensors are
*co-boresighted* — a target at pixel (x, y) in one stream sits at the same pixel
in the other — so detections share a frame and IoU is meaningful directly. Real
hardware often violates this: the two cameras differ in position **and field of
view**, so the same target lands at different pixels, IoU is 0, and nothing ever
fuses.

This module recovers the missing transform automatically, from the detections
themselves — no checkerboard, no manual point picking:

1. On frames where exactly **one** confident EO box and **one** confident IR box
   exist, the pairing is unambiguous (they must be the same object), so we can
   record a correspondence ``(EO center -> IR center)`` *without* already having
   a transform. This breaks the chicken-and-egg of "need a transform to
   associate, need associations to fit a transform".
2. We accumulate these correspondences and, once they cover enough **2-D spread**
   (a target that barely moved cannot constrain scale/rotation — the fit would be
   degenerate), fit an **affine** transform with a small RANSAC loop to reject the
   odd mismatch (e.g. a bird paired with a drone).
3. If the fit's residual is small enough we **lock** it. Because the optics are
   fixed, one lock holds for the whole deployment; re-running only matters if the
   zoom/mount changes.

While unlocked the registrar still *scales* EO boxes into the IR pixel grid, but
without alignment they will not overlap IR — so the pipeline simply runs
un-fused (single-sensor) during this brief warm-up, then fuses once locked.

Pure numpy (an affine fit is linear least-squares); no OpenCV dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .schema import Detection

__all__ = [
    "AutoRegistrar", "RegistrationConfig",
    "fit_affine_lstsq", "ransac_affine",
    "fit_homography_dlt", "ransac_homography",
]


# ---------------------------------------------------------------------------
# Core affine math (numpy only)
# ---------------------------------------------------------------------------
def fit_affine_lstsq(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 2x3 affine mapping ``src`` (Nx2) onto ``dst`` (Nx2).

    Solves ``dst ≈ A @ [x, y, 1]^T`` for the 6 affine parameters. Needs >= 3
    non-collinear point pairs.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = src.shape[0]
    # Design matrix: each point contributes two rows (x and y equations).
    M = np.zeros((2 * n, 6), np.float64)
    M[0::2, 0] = src[:, 0]; M[0::2, 1] = src[:, 1]; M[0::2, 2] = 1.0
    M[1::2, 3] = src[:, 0]; M[1::2, 4] = src[:, 1]; M[1::2, 5] = 1.0
    b = dst.reshape(-1)
    params, *_ = np.linalg.lstsq(M, b, rcond=None)
    return params.reshape(2, 3)


def apply_affine(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to Nx2 points."""
    pts = np.asarray(pts, np.float64)
    return (A[:, :2] @ pts.T + A[:, 2:3]).T


def fit_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 3x3 homography mapping ``src`` (Nx2) onto ``dst`` (Nx2).

    Normalised DLT (Hartley): condition the points, solve ``A h = 0`` by SVD,
    de-normalise. A homography has 8 DOF, so it captures perspective the 6-DOF
    affine cannot. Needs >= 4 non-collinear point pairs.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)

    def _normalize(pts):
        c = pts.mean(0)
        d = pts - c
        scale = np.sqrt(2.0) / (np.sqrt((d ** 2).sum(1)).mean() + 1e-12)
        T = np.array([[scale, 0, -scale * c[0]],
                      [0, scale, -scale * c[1]],
                      [0, 0, 1.0]])
        ptsh = (T @ np.c_[pts, np.ones(len(pts))].T).T
        return ptsh[:, :2], T

    s_n, Ts = _normalize(src)
    d_n, Td = _normalize(dst)
    rows = []
    for (x, y), (u, v) in zip(s_n, d_n):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(rows, np.float64))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to Nx2 points (perspective divide)."""
    pts = np.asarray(pts, np.float64)
    w = (H @ np.c_[pts, np.ones(len(pts))].T).T
    return w[:, :2] / w[:, 2:3]


def _transform(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply either a 2x3 affine or a 3x3 homography (dispatch on shape)."""
    return apply_homography(M, pts) if M.shape == (3, 3) else apply_affine(M, pts)


def _residuals(A: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.linalg.norm(_transform(A, src) - dst, axis=1)


def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    thresh: float = 8.0,
    iters: int = 500,
    seed: int = 0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Robust homography fit. Returns ``(H, inlier_mask)`` or ``(None, ...)``.

    Same RANSAC loop as :func:`ransac_affine` but samples 4 correspondences and
    fits a homography, so it can model perspective between the two views.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = src.shape[0]
    if n < 4:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, bool)
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            H = fit_homography_dlt(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(H)):
            continue
        inliers = _residuals(H, src, dst) < thresh
        c = int(inliers.sum())
        if c > best_count:
            best_count, best_inliers = c, inliers
    if best_count < 4:
        return None, best_inliers
    H = fit_homography_dlt(src[best_inliers], dst[best_inliers])  # refit on consensus
    return H, best_inliers


def ransac_affine(
    src: np.ndarray,
    dst: np.ndarray,
    thresh: float = 8.0,
    iters: int = 200,
    seed: int = 0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Robust affine fit. Returns ``(A, inlier_mask)`` or ``(None, ...)``.

    Samples 3 correspondences, fits, counts inliers within ``thresh`` pixels,
    keeps the largest consensus set, then refits on all its inliers.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = src.shape[0]
    if n < 3:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, bool)
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            A = fit_affine_lstsq(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        inliers = _residuals(A, src, dst) < thresh
        c = int(inliers.sum())
        if c > best_count:
            best_count, best_inliers = c, inliers
    if best_count < 3:
        return None, best_inliers
    A = fit_affine_lstsq(src[best_inliers], dst[best_inliers])  # refit on consensus
    return A, best_inliers


# ---------------------------------------------------------------------------
# Configuration + the self-calibrating registrar
# ---------------------------------------------------------------------------
@dataclass
class RegistrationConfig:
    """Knobs for :class:`AutoRegistrar` (sensible defaults; tune per rig)."""

    ir_size: Tuple[int, int] = (1280, 1024)   # (w, h) target coordinate frame = IR
    model: str = "affine"                      # "affine" (6-DOF) or "homography" (8-DOF)
    conf_min: float = 0.30                     # min confidence to trust a single box
    min_pairs: int = 40                        # need at least this many correspondences
    min_minor_std: float = 12.0                # spread gate: std along the SHORTER axis (px)
    min_major_std: float = 40.0               # spread gate: std along the LONGER axis (px)
    ransac_thresh: float = 8.0                 # inlier distance (px)
    max_residual: float = 8.0                  # lock only if median inlier residual <= this (px)
    min_inlier_frac: float = 0.6               # and at least this fraction are inliers
    max_pairs: int = 4000                      # cap the buffer (keep the most recent)
    refit_every: int = 20                      # after lock, refit once this many new pairs arrive
    settle_pairs: int = 600                    # freeze (stop refitting) past this many correspondences


@dataclass
class AutoRegistrar:
    """Accumulate EO/IR co-detections, fit + lock an EO->IR affine on its own.

    Typical use inside the pipeline, per frame::

        reg.observe(eo_dets, ir_dets, eo_shape)      # learn (no-op once locked)
        eo_in_ir = [reg.apply(d.bbox_xyxy, eo_shape) for d in eo_dets]

    ``apply`` always at least scales the EO box into the IR pixel grid; once
    ``locked`` it additionally warps by the fitted affine so the boxes line up.
    """

    cfg: RegistrationConfig = field(default_factory=RegistrationConfig)
    A: Optional[np.ndarray] = None             # 2x3 affine, EO-scaled -> IR (None until locked)
    locked: bool = False                        # a usable transform exists (fusion may proceed)
    frozen: bool = False                        # stopped refining (settled or loaded from disk)
    last_residual: float = float("nan")
    n_inliers: int = 0
    _src: List[Tuple[float, float]] = field(default_factory=list)   # EO centers (IR-scaled)
    _dst: List[Tuple[float, float]] = field(default_factory=list)   # IR centers
    _since_refit: int = 0

    # -- learning ------------------------------------------------------------
    def observe(self, eo_dets: Sequence[Detection], ir_dets: Sequence[Detection],
                eo_shape: Tuple[int, int]) -> None:
        """Feed one frame. Records a correspondence only from unambiguous frames.

        Keeps refining after the first lock — a static rig only benefits from more
        of the target's trajectory — until ``settle_pairs`` correspondences have
        been seen, then freezes.
        """
        if self.frozen:
            return
        if len(eo_dets) != 1 or len(ir_dets) != 1:
            return                                  # ambiguous: can't pair without a transform
        e, r = eo_dets[0], ir_dets[0]
        if e.conf < self.cfg.conf_min or r.conf < self.cfg.conf_min:
            return
        eh, ew = eo_shape[:2]
        irw, irh = self.cfg.ir_size
        ex1, ey1, ex2, ey2 = e.bbox_xyxy
        sx, sy = irw / ew, irh / eh                 # scale EO native -> IR pixel grid
        self._src.append((float((ex1 + ex2) / 2 * sx), float((ey1 + ey2) / 2 * sy)))
        rx1, ry1, rx2, ry2 = r.bbox_xyxy
        self._dst.append((float((rx1 + rx2) / 2), float((ry1 + ry2) / 2)))
        if len(self._src) > self.cfg.max_pairs:     # keep most recent
            self._src.pop(0); self._dst.pop(0)

        # Fit/refit: every frame until the first lock, then every refit_every pairs.
        self._since_refit += 1
        if not self.locked or self._since_refit >= self.cfg.refit_every:
            self._since_refit = 0
            self._try_lock()
        if self.locked and self.n_pairs >= self.cfg.settle_pairs:
            self.frozen = True                      # enough trajectory; stop refitting

    def _spread_ok(self, src: np.ndarray) -> bool:
        """Reject degenerate (clustered / collinear) point sets via covariance axes."""
        if src.shape[0] < self.cfg.min_pairs:
            return False
        evals = np.linalg.eigvalsh(np.cov(src.T))   # variances along principal axes
        minor, major = float(np.sqrt(max(evals[0], 0))), float(np.sqrt(max(evals[1], 0)))
        return minor >= self.cfg.min_minor_std and major >= self.cfg.min_major_std

    def _try_lock(self) -> None:
        src = np.array(self._src, np.float64)
        dst = np.array(self._dst, np.float64)
        if not self._spread_ok(src):
            return
        if self.cfg.model == "homography":
            A, inliers = ransac_homography(src, dst, thresh=self.cfg.ransac_thresh)
        else:
            A, inliers = ransac_affine(src, dst, thresh=self.cfg.ransac_thresh)
        if A is None:
            return
        frac = inliers.mean()
        resid = float(np.median(_residuals(A, src[inliers], dst[inliers])))
        if frac >= self.cfg.min_inlier_frac and resid <= self.cfg.max_residual:
            self.A, self.locked = A, True
            self.last_residual, self.n_inliers = resid, int(inliers.sum())

    # -- applying ------------------------------------------------------------
    def apply(self, box_xyxy, eo_shape: Tuple[int, int]) -> np.ndarray:
        """EO native box -> IR pixel grid (scaled), warped by the affine if locked."""
        eh, ew = eo_shape[:2]
        irw, irh = self.cfg.ir_size
        sx, sy = irw / ew, irh / eh
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        sb = np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], np.float32)
        if self.A is None:
            return sb                               # warm-up: scaled, not yet aligned
        corners = np.array([[sb[0], sb[1]], [sb[2], sb[1]],
                            [sb[2], sb[3]], [sb[0], sb[3]]], np.float64)
        w = _transform(self.A, corners)   # affine (2x3) or homography (3x3)
        return np.array([w[:, 0].min(), w[:, 1].min(),
                         w[:, 0].max(), w[:, 1].max()], np.float32)

    # -- persistence ---------------------------------------------------------
    @property
    def n_pairs(self) -> int:
        return len(self._src)

    def save(self, path: str) -> None:
        if self.A is None:
            raise RuntimeError("registrar not locked yet; nothing to save")
        np.save(path, self.A)

    def load(self, path: str) -> None:
        """Load a pre-fit affine and lock immediately (skip the warm-up)."""
        self.A = np.load(path)
        self.locked = True
