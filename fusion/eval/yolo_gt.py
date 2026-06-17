"""Load YOLO-format ground truth into pixel xyxy boxes.

YOLO label files are one row per object: ``class_id cx cy w h`` with the box
normalised to [0, 1]. The fused detections live in the shared (EO) pixel frame,
so GT is scaled by that frame's width/height. An image with no matching label
file is a legitimate background image (no objects) and yields an empty list.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

__all__ = ["load_yolo_labels", "label_path_for"]


def label_path_for(image_path: str, labels_dir: str) -> str:
    """Map an image path to its YOLO label .txt in ``labels_dir`` (by basename)."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, base + ".txt")


def load_yolo_labels(label_path: str, img_w: int, img_h: int) -> List[Tuple[np.ndarray, int]]:
    """Return [(bbox_xyxy_pixels, class_id), ...]; empty if file is missing."""
    gts: List[Tuple[np.ndarray, int]] = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h
            gts.append((np.array([x1, y1, x2, y2], dtype=np.float32), c))
    return gts
