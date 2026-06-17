#!/usr/bin/env python3
"""Pre-fit the EO->IR affine from paired ground truth (numpy only, no models).

The pipeline's AutoRegistrar can self-calibrate from detections during a warm-up,
but if you already have paired GT (an EO box and an IR box for the same target in
the same frame) you can fit the transform directly and skip the warm-up entirely.
Save the result and pass it to the eval / pipeline via ``--reg-matrix``.

The affine maps EO centres — scaled into the IR pixel grid — onto IR centres,
which is exactly the convention AutoRegistrar.apply()/load() expect, so
``ir-size`` MUST equal the real IR image size.

    python fusion/eval/prefit_registration.py \
        --eo-labels  DATASETv3/fusion_testset/labels \
        --ir-labels  DATASETv3/fusion_testset/labels_ir \
        --ir-size 1280 1024 \
        --out DATASETv3/fusion_testset/eo2ir_affine.npy
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.registration import ransac_affine, ransac_homography, _residuals  # noqa: E402


def centers(label_path):
    """Return list of (cx, cy) normalised centres in a YOLO label file."""
    out = []
    if not os.path.exists(label_path):
        return out
    for line in open(label_path):
        p = line.split()
        if len(p) >= 5:
            out.append((float(p[1]), float(p[2])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eo-labels", required=True, help="EO YOLO label dir (_0_)")
    ap.add_argument("--ir-labels", required=True, help="IR YOLO label dir (_1_)")
    ap.add_argument("--ir-size", nargs=2, type=int, default=[1280, 1024], metavar=("W", "H"))
    ap.add_argument("--model", choices=["affine", "homography"], default="affine",
                    help="transform model: affine (6-DOF) or homography (8-DOF)")
    ap.add_argument("--ransac-thresh", type=float, default=8.0, help="inlier distance (px)")
    ap.add_argument("--out", required=True, help="output .npy affine path")
    args = ap.parse_args()

    irw, irh = args.ir_size
    src, dst = [], []
    for eo_lbl in sorted(glob.glob(os.path.join(args.eo_labels, "*.txt"))):
        base = os.path.basename(eo_lbl)
        ir_lbl = os.path.join(args.ir_labels, base.replace("_0_", "_1_"))
        eb, rb = centers(eo_lbl), centers(ir_lbl)
        if len(eb) == 1 and len(rb) == 1:                      # unambiguous pairing
            src.append((eb[0][0] * irw, eb[0][1] * irh))       # EO centre in IR grid
            dst.append((rb[0][0] * irw, rb[0][1] * irh))       # IR centre
    src, dst = np.array(src), np.array(dst)
    if len(src) < 3:
        print(f"ERROR: only {len(src)} usable correspondences", file=sys.stderr)
        return 1

    naive = np.median(np.linalg.norm(src - dst, axis=1))
    if args.model == "homography":
        A, inliers = ransac_homography(src, dst, thresh=args.ransac_thresh, iters=500)
    else:
        A, inliers = ransac_affine(src, dst, thresh=args.ransac_thresh, iters=500)
    if A is None:
        print(f"ERROR: RANSAC found no consensus {args.model} — sensors may not be {args.model}-related",
              file=sys.stderr)
        return 1

    resid = _residuals(A, src[inliers], dst[inliers])
    print(f"correspondences      : {len(src)} unambiguous frames")
    print(f"identity (no reg)    : median center error {naive:.1f}px")
    print(f"{args.model} inliers   : {inliers.mean():.0%} ({int(inliers.sum())}/{len(src)})")
    print(f"{args.model} residual  : median {np.median(resid):.2f}px, max {resid.max():.1f}px")
    np.save(args.out, A)
    print(f"saved affine         : {args.out}")
    locks = inliers.mean() >= 0.6 and np.median(resid) <= 8.0
    print(f"would lock (>=60% inliers, <=8px): {'YES' if locks else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
