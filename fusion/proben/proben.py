"""ProbEn — Probabilistic Ensembling for multimodal detection fusion.

Reference:
    Chen, Y.-T., Shi, J., Ye, Z., Mertz, C., Ramanan, D., Kong, S.
    "Multimodal Object Detection via Probabilistic Ensembling", ECCV 2022.

This is a *late* fusion method (same family as the plan's noisy-OR pipeline)
but with a different, fully-Bayesian score rule. Given detections of the same
object from K modalities that are assumed conditionally independent given the
category, the fused posterior over classes is the (prior-corrected) **product**
of the per-modality class distributions:

    p(y | d_1..d_K) ∝ p(y)^(1-K) · Π_k p(y | d_k)

For a detector that emits a single foreground score ``s`` for class ``c`` we use
the categorical distribution ``[..., s at c, ..., (1-s) at background]``. With a
uniform prior the rule reduces to the normalised elementwise product — e.g. two
sensors agreeing at 0.8 fuse to 0.8·0.8 / (0.8·0.8 + 0.2·0.2) = 0.94. Boxes are
fused by score-weighted averaging ("s-avg" in the paper).

Simplifications vs. the paper:
- top-1 score → {class, background} distribution (the paper can use full
  softmax logits when the detector exposes them; pass ``probs=`` per detection
  to use that path).
- per-modality NMS is assumed already done (YOLO output is).

Depends only on numpy. Consumes the pipeline's ``Detection`` and emits
``FusedDetection`` so it is a drop-in alternative to ``decision_logic``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Make the sibling ``pipeline`` package importable when used standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.schema import Detection, FusedDetection  # noqa: E402

__all__ = ["ProbEnConfig", "proben_fuse", "proben_two"]


@dataclass
class ProbEnConfig:
    num_classes: int = 1
    assoc_iou: float = 0.5                 # IoU to group detections across modalities
    box_fusion: str = "s-avg"              # "s-avg" (score-weighted) | "avg"
    prior: Optional[np.ndarray] = None     # length num_classes+1 (last=bg); None=uniform
    score_floor: float = 1e-4              # clamp scores away from 0/1
    min_score: float = 0.0                 # drop fused dets below this (0 = keep all)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _distribution(det: Detection, num_classes: int, floor: float) -> np.ndarray:
    """Categorical [class_0..class_{C-1}, background] from a top-1 detection."""
    p = np.full(num_classes + 1, floor, dtype=np.float64)
    s = float(np.clip(det.conf, floor, 1.0 - floor))
    cid = det.class_id if 0 <= det.class_id < num_classes else 0
    p[cid] = s
    p[num_classes] = 1.0 - s   # background
    return p


def _cluster(det_lists: Sequence[Sequence[Detection]], assoc_iou: float):
    """Group detections across modalities; <=1 detection per modality per group.

    Greedy, seeded from the highest-confidence detection (NMS-like grouping).
    Each group is a dict {modality_index: Detection}.
    """
    items = [(m, d) for m, lst in enumerate(det_lists) for d in lst]
    items.sort(key=lambda md: md[1].conf, reverse=True)
    used = [False] * len(items)
    groups = []
    for i, (m, d) in enumerate(items):
        if used[i]:
            continue
        used[i] = True
        group = {m: d}
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            m2, d2 = items[j]
            if m2 in group:
                continue
            if _iou(d.bbox_xyxy, d2.bbox_xyxy) >= assoc_iou:
                group[m2] = d2
                used[j] = True
        groups.append(group)
    return groups


def _fuse_box(group, mode: str) -> np.ndarray:
    boxes = np.array([d.bbox_xyxy for d in group.values()], dtype=np.float64)
    if mode == "avg" or len(boxes) == 1:
        return boxes.mean(axis=0)
    w = np.array([d.conf for d in group.values()], dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    return (boxes * w[:, None]).sum(axis=0)


def proben_fuse(
    det_lists: Sequence[Sequence[Detection]],
    source_labels: Sequence[str],
    config: Optional[ProbEnConfig] = None,
    regime: str = "",
) -> List[FusedDetection]:
    """Probabilistically ensemble detections from K modalities (plan-agnostic).

    ``det_lists[k]`` are the detections from modality ``source_labels[k]``.
    Returns fused detections with the Bayesian posterior as ``conf``.
    """
    cfg = config or ProbEnConfig()
    C = cfg.num_classes
    prior = cfg.prior if cfg.prior is not None else np.full(C + 1, 1.0 / (C + 1))

    out: List[FusedDetection] = []
    for group in _cluster(det_lists, cfg.assoc_iou):
        k_eff = len(group)

        # Product of experts with prior correction: p(y) ∝ prior^(1-K) Π_k p_k.
        log_post = (1.0 - k_eff) * np.log(prior)
        for d in group.values():
            log_post = log_post + np.log(_distribution(d, C, cfg.score_floor))
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()

        fg = post[:C]
        cls = int(np.argmax(fg))
        score = float(fg[cls])
        if score < cfg.min_score:
            continue

        out.append(
            FusedDetection(
                bbox_xyxy=_fuse_box(group, cfg.box_fusion),
                conf=score,
                support=sorted(source_labels[m] for m in group),
                regime=regime,
                class_id=cls,
            )
        )
    return out


def proben_two(
    eo_dets: Sequence[Detection],
    ir_dets: Sequence[Detection],
    config: Optional[ProbEnConfig] = None,
    regime: str = "",
) -> List[FusedDetection]:
    """Convenience wrapper for the two-sensor (EO, IR) case."""
    return proben_fuse([eo_dets, ir_dets], ["EO", "IR"], config, regime)
