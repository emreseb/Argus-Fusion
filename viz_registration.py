#!/usr/bin/env python3
"""Show EO boxes registered into the IR frame.

For frames with one drone in each sensor, warp the EO ground-truth box into the
IR frame with the fitted affine and draw it on the IR image (green), next to the
IR ground-truth box (red). Builds a montage so you can eyeball the alignment.
"""
import glob
import os

import cv2
import numpy as np

FT = "/home/emre/Desktop/NATO/DATASETv3/fusion_testset"
A = np.load(os.path.join(FT, "eo2ir_affine.npy"))


def yolo_to_xyxy(cx, cy, bw, bh, W, H):
    return [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]


def read_label(path):
    if not os.path.exists(path):
        return None
    lines = [l.split() for l in open(path) if len(l.split()) >= 5]
    return [tuple(map(float, l[1:5])) for l in lines]


def register_eo_box(box_eo_xyxy, eo_w, eo_h, ir_w, ir_h):
    sx, sy = ir_w / eo_w, ir_h / eo_h
    x1, y1, x2, y2 = box_eo_xyxy
    sb = np.array([[x1 * sx, y1 * sy], [x2 * sx, y1 * sy],
                   [x2 * sx, y2 * sy], [x1 * sx, y2 * sy]])
    w = (A[:, :2] @ sb.T + A[:, 2:3]).T
    return [w[:, 0].min(), w[:, 1].min(), w[:, 0].max(), w[:, 1].max()]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    eo_lbls = sorted(glob.glob(os.path.join(FT, "labels", "*.txt")))
    picks = []
    for eo_lbl in eo_lbls:
        base = os.path.basename(eo_lbl)
        ir_lbl = os.path.join(FT, "labels_ir", base.replace("_0_", "_1_"))
        eb, rb = read_label(eo_lbl), read_label(ir_lbl)
        if eb and rb and len(eb) == 1 and len(rb) == 1:
            picks.append((base[:-4], eb[0], rb[0]))
    # spread 6 samples across the dataset
    idxs = np.linspace(0, len(picks) - 1, 6).astype(int)
    tiles = []
    for i in idxs:
        stem, eo_box_n, ir_box_n = picks[i]
        eo_img = cv2.imread(os.path.join(FT, "eo", stem + ".jpg"))
        ir_img = cv2.imread(os.path.join(FT, "ir", stem.replace("_0_", "_1_") + ".jpg"))
        eh, ew = eo_img.shape[:2]
        ih, iw = ir_img.shape[:2]
        eo_xyxy = yolo_to_xyxy(*eo_box_n, ew, eh)
        ir_xyxy = yolo_to_xyxy(*ir_box_n, iw, ih)
        reg = register_eo_box(eo_xyxy, ew, eh, iw, ih)
        canvas = ir_img.copy()
        # IR ground truth = red ; registered EO = green
        cv2.rectangle(canvas, (int(ir_xyxy[0]), int(ir_xyxy[1])), (int(ir_xyxy[2]), int(ir_xyxy[3])), (0, 0, 255), 2)
        cv2.rectangle(canvas, (int(reg[0]), int(reg[1])), (int(reg[2]), int(reg[3])), (0, 255, 0), 2)
        cv2.putText(canvas, f"IoU={iou(reg, ir_xyxy):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(cv2.resize(canvas, (480, 384)))
    montage = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    cv2.putText(montage, "green = EO box registered into IR   |   red = IR ground truth",
                (10, montage.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out = "/home/emre/Desktop/NATO/code/runs/reg_viz.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, montage)
    print("usable single-drone pairs:", len(picks))
    print("saved ->", out)


if __name__ == "__main__":
    main()
