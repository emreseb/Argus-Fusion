#!/usr/bin/env python3
"""EO/IR late fusion with affine registration (EO -> IR coordinate space).

The two sensors are NOT co-boresighted: their fields of view differ in position
AND scale, so a resize/crop cannot line up detections (measured: a translation
leaves ~97px median error, while one affine leaves ~5px). This script therefore
registers each EO detection into the IR image frame with a fixed affine fitted
from the drone's trajectory in both views, then runs the existing fusion logic.

    EO frame --(YOLO)--> EO boxes --(scale to IR size)--(affine A)--> IR space
    IR frame --(YOLO)--> IR boxes (already IR space)
    -> FusionPipeline.process_detections -> fused boxes + tracks (drawn on IR)

Controls with --show --step: -> / d = next frame, <- / a = previous frame
(reversible: frames are cached so you can scrub back and forth), 'q'/Esc = quit.
Without --step the window auto-plays forward.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "fusion"))

from ultralytics import YOLO  # noqa: E402
from pipeline import FusionPipeline, PipelineConfig, Detection  # noqa: E402
from pipeline.viz import draw_result  # noqa: E402

EO_MODEL = os.path.join(_ROOT, "fusion/models/eo/EO-150-epoch.pt")
IR_MODEL = os.path.join(_ROOT, "fusion/models/ir/IR-150-epoch.pt")
EO_VID = "/home/emre/Desktop/NATO/vid-src/EO-stream.mp4"
IR_VID = "/home/emre/Desktop/NATO/vid-src/IR-stream.mp4"
IRW, IRH = 1280, 1024


def warp_box(box, sx, sy, A):
    """EO native box -> scale to IR size -> affine -> axis-aligned IR box."""
    x1, y1, x2, y2 = box
    corners = np.array([[x1 * sx, y1 * sy], [x2 * sx, y1 * sy],
                        [x2 * sx, y2 * sy], [x1 * sx, y2 * sy]], np.float32)
    w = (A[:, :2] @ corners.T + A[:, 2:3]).T
    return [w[:, 0].min(), w[:, 1].min(), w[:, 0].max(), w[:, 1].max()]


def mean_v(frame_bgr):
    small = cv2.resize(frame_bgr, (256, 256))
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2].mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--affine", default=os.path.join(_ROOT, "fusion/out/affine_eo2ir.npy"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    A = np.load(args.affine)
    eo_model, ir_model = YOLO(EO_MODEL), YOLO(IR_MODEL)
    pipe = FusionPipeline(PipelineConfig())  # default DAY/TWILIGHT/NIGHT params

    eo_cap, ir_cap = cv2.VideoCapture(EO_VID), cv2.VideoCapture(IR_VID)
    win = "EO->IR registered fusion"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, IRW, IRH)

    n_eo = int(eo_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_ir = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_total = min(n_eo, n_ir)
    if args.max_frames:
        n_total = min(n_total, args.max_frames)

    # The Kalman tracker is stateful, so each frame must be processed exactly
    # once, in increasing order. We cache the FrameResult per index; revisiting
    # a frame (scrubbing back) just re-draws the cache, never re-runs the models
    # or the tracker. New work only happens at the frontier (highest index + 1).
    cache: dict = {}
    frontier = [-1]

    def read_pair(idx):
        eo_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ir_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok_e, eo = eo_cap.read()
        ok_r, ir = ir_cap.read()
        return (eo if ok_e else None), (ir if ok_r else None)

    def compute(idx, eo, ir):
        eh, ew = eo.shape[:2]
        sx, sy = IRW / ew, IRH / eh
        re = eo_model.predict(eo, conf=args.conf, verbose=False)[0]
        ri = ir_model.predict(ir, conf=args.conf, verbose=False)[0]
        eo_dets = [Detection("EO", warp_box(b.tolist(), sx, sy, A), float(c), int(cl))
                   for b, c, cl in zip(re.boxes.xyxy, re.boxes.conf, re.boxes.cls)]
        ir_dets = [Detection("IR", b.tolist(), float(c), int(cl))
                   for b, c, cl in zip(ri.boxes.xyxy, ri.boxes.conf, ri.boxes.cls)]
        return pipe.process_detections(eo_dets, ir_dets, regime=mean_v(eo), frame_ts=idx / 30.0)

    def log_line(idx, result):
        fused = ", ".join(f"{'+'.join(f.support)}:{f.conf:.2f}" for f in result.fused) or "-"
        trk = ", ".join(f"#{t.id}" for t in result.tracks) or "-"
        eo_n = sum(1 for d in result.eo_dets) if hasattr(result, "eo_dets") else 0
        ir_n = sum(1 for d in result.ir_dets) if hasattr(result, "ir_dets") else 0
        print(f"[{idx:05d}] {result.regime.value:8s} EO={eo_n} IR={ir_n} "
              f"fused[{fused}] tracks[{trk}]")

    # Arrow key codes differ across OpenCV GUI backends (Qt/GTK/Win); accept all.
    LEFT = {65361, 81, 2424832, ord('a'), ord('h')}
    RIGHT = {65363, 83, 2555904, ord('d'), ord('l'), ord(' '), 13}
    QUIT = {ord('q'), 27}

    writer = None
    n_fused = n_track = 0

    if args.show:
        idx = 0
        while 0 <= idx < n_total:
            eo, ir = read_pair(idx)
            if ir is None:
                idx = max(0, idx - 1)
                continue
            if idx not in cache:
                if eo is None:
                    break
                cache[idx] = compute(idx, eo, ir)
                frontier[0] = idx
                if args.do_print:
                    log_line(idx, cache[idx])
            result = cache[idx]
            vis = draw_result(ir, result, pipe.cfg.class_names)
            hud = f"frame {idx}/{n_total - 1}   [<-]/a prev   [->]/d next   q quit"
            cv2.putText(vis, hud, (10, vis.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, vis)
            key = cv2.waitKeyEx(0 if args.step else 1)
            if key in QUIT or (key & 0xFF) in QUIT:
                break
            if not args.step:
                idx = idx + 1
            elif key in LEFT or (key & 0xFF) in {ord('a'), ord('h')}:
                idx = max(0, idx - 1)
            elif key in RIGHT or (key & 0xFF) in {ord('d'), ord('l'), ord(' ')}:
                idx = min(n_total - 1, idx + 1)
            # any other key: stay on the current frame and redraw
        n_fused = sum(1 for r in cache.values() if r.fused)
        n_track = sum(1 for r in cache.values() if r.tracks)
        cv2.destroyAllWindows()
    else:
        # Headless / --save: stream sequentially (no seeking), process once each.
        eo_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ir_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while idx < n_total:
            ok_e, eo = eo_cap.read()
            ok_r, ir = ir_cap.read()
            if not (ok_e and ok_r):
                break
            result = compute(idx, eo, ir)
            if result.fused:
                n_fused += 1
            if result.tracks:
                n_track += 1
            if args.do_print:
                log_line(idx, result)
            if args.save:
                vis = draw_result(ir, result, pipe.cfg.class_names)
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
                writer.write(vis)
            idx += 1

    eo_cap.release(); ir_cap.release()
    if writer is not None:
        writer.release()
    print(f"Processed {len(cache) if args.show else idx} frames | "
          f"{n_fused} with fused det | {n_track} with track")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
