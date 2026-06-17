#!/usr/bin/env python3
"""Side-by-side EO|IR detector inspection (no fusion, no registration).

Each model runs on its OWN native frame and draws its OWN boxes there, so what
you see is exactly what each detector decides in its own image. The EO panel is
then scaled to the IR frame size so the two panels sit the same size next to
each other (left = EO, right = IR).

Controls (focus the window): any key = next frame, 'q' = quit.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

EO_MODEL = "fusion/models/eo/EO-150-epoch.pt"
IR_MODEL = "fusion/models/ir/IR-150-epoch.pt"
EO_VID = "/home/emre/Desktop/NATO/vid-src/EO-stream.mp4"
IR_VID = "/home/emre/Desktop/NATO/vid-src/IR-stream.mp4"


def banner(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--step", action="store_true", help="pause each frame (any key=next, q=quit)")
    ap.add_argument("--show", action="store_true", help="display a window")
    ap.add_argument("--save", help="write the side-by-side video here")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    eo_model, ir_model = YOLO(EO_MODEL), YOLO(IR_MODEL)
    eo_cap, ir_cap = cv2.VideoCapture(EO_VID), cv2.VideoCapture(IR_VID)
    ir_w = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ir_h = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # panel size = IR dims

    win = "EO (left)  |  IR (right)"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 2 * ir_w // 2, ir_h // 2)  # half-scale window to fit screen

    writer = None
    i = 0
    while True:
        ok_e, eo = eo_cap.read()
        ok_r, ir = ir_cap.read()
        if not (ok_e and ok_r):
            break

        re = eo_model.predict(eo, conf=args.conf, verbose=False)[0]
        ri = ir_model.predict(ir, conf=args.conf, verbose=False)[0]
        eo_vis = cv2.resize(re.plot(), (ir_w, ir_h))    # scale whole EO to IR size
        ir_vis = ri.plot()
        banner(eo_vis, f"EO  f{i}  det={len(re.boxes)}")
        banner(ir_vis, f"IR  f{i}  det={len(ri.boxes)}")
        combo = np.hstack([eo_vis, ir_vis])

        if args.save:
            if writer is None:
                h, w = combo.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
            writer.write(combo)
        if args.show:
            cv2.imshow(win, combo)
            if (cv2.waitKey(0 if args.step else 1) & 0xFF) == ord("q"):
                break

        i += 1
        if args.max_frames and i >= args.max_frames:
            break

    eo_cap.release(); ir_cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Processed {i} frame pair(s).")
    return 0


if __name__ == "__main__":
    # Resolve model paths relative to the repo root (parent of this file's dir).
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EO_MODEL = os.path.join(_root, EO_MODEL)
    IR_MODEL = os.path.join(_root, IR_MODEL)
    raise SystemExit(main())
