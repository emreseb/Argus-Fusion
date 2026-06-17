"""Self-contained detection metrics (numpy only).

Computes precision / recall / F1 / AP@0.5 / AP@0.75 / mAP@[.5:.95] from
prediction and ground-truth box lists, so the fusion eval doesn't depend on
pycocotools. AP uses the standard 101-point interpolation (COCO-style); the
operating-point P/R/F1 are reported at the confidence that maximises F1.

Data format (per image):
    preds : list of (bbox_xyxy: np.ndarray[4], score: float, class_id: int)
    gts   : list of (bbox_xyxy: np.ndarray[4], class_id: int)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["evaluate", "IOU_SWEEP"]

IOU_SWEEP = np.round(np.arange(0.5, 1.0, 0.05), 2)  # 0.5, 0.55, ... 0.95


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (M x4) and b (N x4) -> (M x N)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _match_image(preds, gts, iou_thr: float, cls: int):
    """Greedy score-ordered matching for one image and one class.

    Returns (scores, tps) for this class's predictions plus the gt count.
    """
    p = [(b, s) for (b, s, c) in preds if c == cls]
    g = np.array([b for (b, c) in gts if c == cls], dtype=np.float32)
    n_gt = len(g)
    if not p:
        return np.zeros(0), np.zeros(0), n_gt

    p.sort(key=lambda x: x[1], reverse=True)
    pboxes = np.array([b for b, _ in p], dtype=np.float32)
    scores = np.array([s for _, s in p], dtype=np.float32)
    tps = np.zeros(len(p), dtype=np.float32)

    if n_gt:
        ious = _iou_matrix(pboxes, g)
        claimed = np.zeros(n_gt, dtype=bool)
        for i in range(len(p)):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= iou_thr and not claimed[j]:
                tps[i] = 1.0
                claimed[j] = True
    return scores, tps, n_gt


def _ap_from_pr(scores: np.ndarray, tps: np.ndarray, n_gt: int):
    """AP (101-pt interp) plus the full precision/recall arrays."""
    if n_gt == 0:
        return 0.0, None
    if len(scores) == 0:
        return 0.0, (np.array([0.0]), np.array([0.0]), np.array([1.0]))

    order = np.argsort(-scores)
    scores = scores[order]
    tps = tps[order]
    fps = 1.0 - tps

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recall = cum_tp / (n_gt + 1e-12)
    precision = cum_tp / (cum_tp + cum_fp + 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):  # precision envelope
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)  # COCO 101-point interpolation
    y = np.interp(x, mrec, mpre)
    ap = float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))  # trapezoid (np.trapz-free)
    return ap, (scores, recall, precision)


def _best_f1(scores, recall, precision):
    """Operating point (precision, recall, f1, conf) at max F1 along the curve."""
    if recall is None or len(recall) == 0:
        return 0.0, 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    k = int(np.argmax(f1))
    conf = float(scores[k]) if len(scores) else 0.0
    return float(precision[k]), float(recall[k]), float(f1[k]), conf


def evaluate(
    preds_by_image: Sequence[Sequence[Tuple[np.ndarray, float, int]]],
    gts_by_image: Sequence[Sequence[Tuple[np.ndarray, int]]],
    class_ids: Sequence[int],
    class_names: Sequence[str] = None,
) -> Dict:
    """Full evaluation over a dataset.

    Returns a dict with overall + per-class precision/recall/f1/AP50/AP75/
    mAP50-95 and the raw gt counts.
    """
    class_ids = list(class_ids)
    names = {c: (class_names[i] if class_names else str(c)) for i, c in enumerate(class_ids)}

    # mAP@[.5:.95]: AP per class at each IoU threshold.
    ap_by_iou = {c: [] for c in class_ids}
    for thr in IOU_SWEEP:
        for c in class_ids:
            scores_all, tps_all, n_gt = [], [], 0
            for preds, gts in zip(preds_by_image, gts_by_image):
                s, t, ng = _match_image(preds, gts, float(thr), c)
                scores_all.append(s)
                tps_all.append(t)
                n_gt += ng
            scores_all = np.concatenate(scores_all) if scores_all else np.zeros(0)
            tps_all = np.concatenate(tps_all) if tps_all else np.zeros(0)
            ap, _ = _ap_from_pr(scores_all, tps_all, n_gt)
            ap_by_iou[c].append((float(thr), ap))

    # Per-class detailed metrics at IoU 0.5 (for P/R/F1 + AP50) and 0.75.
    per_class = {}
    for c in class_ids:
        row = {"name": names[c]}
        for thr, key in ((0.5, "AP50"), (0.75, "AP75")):
            scores_all, tps_all, n_gt = [], [], 0
            for preds, gts in zip(preds_by_image, gts_by_image):
                s, t, ng = _match_image(preds, gts, thr, c)
                scores_all.append(s)
                tps_all.append(t)
                n_gt += ng
            scores_all = np.concatenate(scores_all) if scores_all else np.zeros(0)
            tps_all = np.concatenate(tps_all) if tps_all else np.zeros(0)
            ap, pr = _ap_from_pr(scores_all, tps_all, n_gt)
            row[key] = ap
            row["n_gt"] = n_gt
            if thr == 0.5:
                if pr is not None:
                    s_sorted, rec, prec = pr
                    p, r, f1, conf = _best_f1(s_sorted, rec, prec)
                else:
                    p = r = f1 = conf = 0.0
                row.update({"precision": p, "recall": r, "f1": f1, "conf": conf})
        row["mAP50-95"] = float(np.mean([ap for _, ap in ap_by_iou[c]])) if ap_by_iou[c] else 0.0
        per_class[c] = row

    # Overall = mean over classes that have ground-truth.
    present = [c for c in class_ids if per_class[c]["n_gt"] > 0]
    def _mean(key):
        return float(np.mean([per_class[c][key] for c in present])) if present else 0.0

    overall = {k: _mean(k) for k in ("precision", "recall", "f1", "AP50", "AP75", "mAP50-95")}
    overall["n_gt"] = sum(per_class[c]["n_gt"] for c in class_ids)
    return {"overall": overall, "per_class": per_class, "classes_present": present}
