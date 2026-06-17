#!/usr/bin/env python3
"""Run the EO/IR fusion pipeline over a pair of synchronised streams.

Each stream is either a video file or a directory of frames. EO and IR frames
are paired by index (the sensors are assumed co-boresighted and frame-synced;
swap in timestamp pairing via pipeline.sync.nearest_timestamp if you have
per-frame timestamps).

Examples
--------
    # two videos, live window
    python fusion/run_fusion.py --eo eo.mp4 --ir ir.mp4 \
        --eo-model best.pt --ir-model bestmodels/best13.pt --show

    # two frame folders, write an annotated mp4, no GPU model needed if --no-sahi
    python fusion/run_fusion.py --eo frames/eo --ir frames/ir --save out.mp4

    # headless: just print fused detections + tracks per frame
    python fusion/run_fusion.py --eo eo.mp4 --ir ir.mp4 --print --max-frames 50
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# Make the sibling ``pipeline`` package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import FusionPipeline, PipelineConfig  # noqa: E402

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def frame_source(path: str):
    """Yield (index, frame_bgr, timestamp) for a video file or image folder."""
    import cv2

    ext = os.path.splitext(path)[1].lower()
    if os.path.isdir(path) or ext not in _VIDEO_EXTS:
        if os.path.isdir(path):
            files = []
            for e in _IMG_EXTS:
                files += glob.glob(os.path.join(path, "**", f"*{e}"), recursive=True)
            files = sorted(files)
        else:  # treat as a glob pattern
            files = sorted(glob.glob(path, recursive=True))
        if not files:
            raise FileNotFoundError(f"No images found for: {path}")
        for i, f in enumerate(files):
            img = cv2.imread(f)
            if img is None:
                print(f"WARN: could not read {f}", file=sys.stderr)
                continue
            yield i, img, float(i)
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame, i / fps
        i += 1
    cap.release()


def build_config(args) -> PipelineConfig:
    if args.config:
        cfg = PipelineConfig.from_json(args.config)
    else:
        cfg = PipelineConfig()
    # CLI overrides win.
    if args.eo_model:
        cfg.eo_model_path = args.eo_model
    if args.ir_model:
        cfg.ir_model_path = args.ir_model
    if args.device:
        cfg.device = args.device
    if args.no_sahi:
        cfg.use_sahi = False
    if args.t_low is not None:
        cfg.t_low = args.t_low
    if args.t_high is not None:
        cfg.t_high = args.t_high
    if args.params:
        cfg.params_path = args.params
    if args.feedback:
        cfg.use_track_feedback = True
    if args.register:
        cfg.use_registration = True
    if args.reg_matrix:
        cfg.use_registration = True
        cfg.registration_matrix_path = args.reg_matrix
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eo", required=True, help="EO video file or frame directory")
    ap.add_argument("--ir", required=True, help="IR video file or frame directory")
    ap.add_argument("--eo-model", help="path to EO weights (.pt)")
    ap.add_argument("--ir-model", help="path to IR weights (.pt)")
    ap.add_argument("--config", help="PipelineConfig JSON")
    ap.add_argument("--params", help="per-regime params JSON (plan §6 overrides)")
    ap.add_argument("--device", help="auto | cpu | cuda:0 | mps")
    ap.add_argument("--no-sahi", action="store_true", help="plain YOLO instead of SAHI slicing")
    ap.add_argument("--feedback", action="store_true", help="enable trajectory feedback (plan §5)")
    ap.add_argument("--register", action="store_true",
                    help="self-calibrating EO->IR affine registration (non-coboresighted sensors)")
    ap.add_argument("--reg-matrix", help="load a pre-fit affine .npy and skip warm-up")
    ap.add_argument("--save-reg", help="after the run, save the locked affine to this .npy")
    ap.add_argument("--t-low", type=float, default=None, help="NIGHT/TWILIGHT brightness threshold")
    ap.add_argument("--t-high", type=float, default=None, help="TWILIGHT/DAY brightness threshold")
    ap.add_argument("--show", action="store_true", help="display an annotated window")
    ap.add_argument("--step", action="store_true",
                    help="with --show, pause on each frame (any key = next, 'q' = quit)")
    ap.add_argument("--save", help="write an annotated video to this path")
    ap.add_argument("--print", dest="do_print", action="store_true", help="print fused/track info")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--fps", type=float, default=30.0, help="fps for --save output")
    args = ap.parse_args()

    cfg = build_config(args)
    pipe = FusionPipeline(cfg)

    import cv2
    from pipeline.viz import draw_result

    eo_iter = frame_source(args.eo)
    ir_iter = frame_source(args.ir)

    writer = None
    win = "EO/IR Fusion"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

    n = 0
    try:
        for (i, eo_img, ts), (_, ir_img, _) in zip(eo_iter, ir_iter):
            result = pipe.process_frame(eo_img, ir_img, frame_ts=ts)

            if args.do_print:
                fused = ", ".join(f"{'+'.join(f.support)}:{f.conf:.2f}" for f in result.fused) or "-"
                trk = ", ".join(f"#{t.id}@({t.bbox[0]:.0f},{t.bbox[1]:.0f})" for t in result.tracks) or "-"
                reg = result.extra.get("registration")
                regs = (f" reg[{'LOCKED' if reg['locked'] else 'warmup'} "
                        f"n={reg['n_pairs']} r={reg['residual_px']:.1f}px]") if reg else ""
                print(f"[{i:05d}] {result.regime.value:8s} meanV={result.mean_v:5.1f} "
                      f"EO={len(result.eo_dets)} IR={len(result.ir_dets)} | fused[{fused}] | tracks[{trk}]{regs}")

            if args.show or args.save:
                # With registration everything lives in the IR frame; draw there.
                canvas = ir_img if cfg.use_registration else eo_img
                vis = draw_result(canvas, result, cfg.class_names)
                if args.save:
                    if writer is None:
                        h, w = vis.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (w, h))
                    writer.write(vis)
                if args.show:
                    cv2.imshow(win, vis)
                    if (cv2.waitKey(0 if args.step else 1) & 0xFF) == ord("q"):
                        break

            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if args.save_reg and pipe.registrar is not None and pipe.registrar.locked:
        pipe.registrar.save(args.save_reg)
        print(f"Saved registration affine -> {args.save_reg} "
              f"(residual {pipe.registrar.last_residual:.1f}px, {pipe.registrar.n_pairs} pairs)")

    print(f"Processed {n} frame pair(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
