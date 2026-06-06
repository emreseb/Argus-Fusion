"""Decision logic — the fusion core (plan §3).

Input: a list of EO detections and a list of IR detections living in a shared
image frame (co-boresighted, no registration), plus the active regime params.
Output: a small set of fused detections, each with a single confidence.

Three sub-steps: associate -> fuse boxes (WBF) -> fuse confidence (noisy-OR),
followed by the agreement bonus / lonely penalty / decision threshold.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .params import RegimeParams, Regime
from .schema import Detection, FusedDetection

__all__ = ["iou", "associate", "wbf", "noisy_or", "decision_logic"]


def iou(box_a, box_b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes that share a frame (plan §3.1)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def associate(
    eo_dets: Sequence[Detection],
    ir_dets: Sequence[Detection],
    assoc_iou: float,
    require_same_class: bool = False,
) -> Tuple[List[Tuple[Detection, Detection]], List[Detection], List[Detection]]:
    """Greedy IoU matching of EO vs IR boxes (plan §3.1).

    Match pairs with ``IoU >= assoc_iou``, highest IoU first. Returns
    ``(pairs, eo_only, ir_only)``:

    - **pairs**     — matched EO+IR (both sensors agree), strongest evidence.
    - **eo_only**   — EO boxes with no IR match.
    - **ir_only**   — IR boxes with no EO match.

    The plan's §3.1 is IoU-only; ``require_same_class`` is an optional guard so
    an EO "drone" is never fused with an IR "bird" at high overlap (they then
    stay as singles, i.e. a disagreement).
    """
    candidates = []  # (iou, i, j)
    for i, e in enumerate(eo_dets):
        for j, r in enumerate(ir_dets):
            if require_same_class and e.class_id != r.class_id:
                continue
            score = iou(e.bbox_xyxy, r.bbox_xyxy)
            if score >= assoc_iou:
                candidates.append((score, i, j))

    candidates.sort(key=lambda c: c[0], reverse=True)

    used_eo, used_ir = set(), set()
    pairs: List[Tuple[Detection, Detection]] = []
    for _, i, j in candidates:
        if i in used_eo or j in used_ir:
            continue
        used_eo.add(i)
        used_ir.add(j)
        pairs.append((eo_dets[i], ir_dets[j]))

    eo_only = [e for i, e in enumerate(eo_dets) if i not in used_eo]
    ir_only = [r for j, r in enumerate(ir_dets) if j not in used_ir]
    return pairs, eo_only, ir_only


def wbf(eo_det: Detection, ir_det: Detection, params: RegimeParams) -> np.ndarray:
    """Weighted Box Fusion of an agreement pair (plan §3.2).

    The decided box is the confidence-and-modality-weighted average of the two
    corner sets — more stable than NMS, which would throw one box away.
    """
    w_eo = params.modality_weight_EO * eo_det.conf
    w_ir = params.modality_weight_IR * ir_det.conf
    total = w_eo + w_ir
    if total <= 0.0:
        # Degenerate (both zero weight): fall back to a plain average.
        return (eo_det.bbox_xyxy + ir_det.bbox_xyxy) / 2.0
    return (w_eo * eo_det.bbox_xyxy + w_ir * ir_det.bbox_xyxy) / total


def noisy_or(*probs: float) -> float:
    """Noisy-OR combination: ``1 - prod(1 - p)`` (plan §3.3, generalised §8).

    Two independent agreeing sensors reinforce each other without letting one
    over-confident sensor saturate the result.
    """
    keep = 1.0
    for p in probs:
        keep *= (1.0 - float(p))
    return 1.0 - keep


def _winning_class(eo_det: Detection, ir_det: Detection, params: RegimeParams) -> int:
    """Class id of the more strongly-weighted sensor in an agreement pair."""
    w_eo = params.modality_weight_EO * eo_det.conf
    w_ir = params.modality_weight_IR * ir_det.conf
    return eo_det.class_id if w_eo >= w_ir else ir_det.class_id


def _apply_prior(
    bbox: np.ndarray,
    conf: float,
    priors: Optional[Sequence[np.ndarray]],
    prior_iou_gate: float,
    prior_bonus: float,
    no_prior_penalty: float,
) -> float:
    """Optional trajectory feedback (plan §5): boost detections that land where
    a track predicted, discount ones that appear from nowhere."""
    if priors is None:
        return conf
    matched = any(iou(bbox, p) >= prior_iou_gate for p in priors)
    if matched:
        return min(1.0, conf * (1.0 + prior_bonus))
    return conf * (1.0 - no_prior_penalty)


def decision_logic(
    eo_dets: Sequence[Detection],
    ir_dets: Sequence[Detection],
    params: RegimeParams,
    regime: Regime,
    require_same_class: bool = False,
    priors: Optional[Sequence[np.ndarray]] = None,
    prior_iou_gate: float = 0.3,
    prior_bonus: float = 0.15,
    no_prior_penalty: float = 0.20,
) -> List[FusedDetection]:
    """Run associate -> WBF -> noisy-OR -> adjustments -> threshold (plan §3.3, §7).

    ``priors`` (predicted track boxes) enables the optional trajectory feedback
    of plan §5; leave it ``None`` to reproduce the plan's §7 pseudocode exactly.
    """
    regime_name = regime.value if isinstance(regime, Regime) else str(regime)
    pairs, eo_only, ir_only = associate(
        eo_dets, ir_dets, params.assoc_iou, require_same_class=require_same_class
    )

    out: List[FusedDetection] = []

    # Agreement pairs — both sensors fired and overlapped.
    for e, r in pairs:
        box = wbf(e, r, params)
        conf = noisy_or(
            params.modality_weight_EO * e.conf,
            params.modality_weight_IR * r.conf,
        )
        conf = min(1.0, conf * (1.0 + params.agreement_bonus))  # agreement bonus
        conf = _apply_prior(box, conf, priors, prior_iou_gate, prior_bonus, no_prior_penalty)
        out.append(
            FusedDetection(
                bbox_xyxy=box,
                conf=conf,
                support=["EO", "IR"],
                regime=regime_name,
                class_id=_winning_class(e, r, params),
            )
        )

    # EO-only — discounted by the regime's EO lonely penalty.
    for e in eo_only:
        conf = params.modality_weight_EO * e.conf * (1.0 - params.lonely_penalty_EO)
        conf = _apply_prior(e.bbox_xyxy, conf, priors, prior_iou_gate, prior_bonus, no_prior_penalty)
        out.append(
            FusedDetection(
                bbox_xyxy=e.bbox_xyxy,
                conf=conf,
                support=["EO"],
                regime=regime_name,
                class_id=e.class_id,
            )
        )

    # IR-only — discounted by the regime's IR lonely penalty.
    for r in ir_only:
        conf = params.modality_weight_IR * r.conf * (1.0 - params.lonely_penalty_IR)
        conf = _apply_prior(r.bbox_xyxy, conf, priors, prior_iou_gate, prior_bonus, no_prior_penalty)
        out.append(
            FusedDetection(
                bbox_xyxy=r.bbox_xyxy,
                conf=conf,
                support=["IR"],
                regime=regime_name,
                class_id=r.class_id,
            )
        )

    return [d for d in out if d.conf >= params.decision_threshold]
