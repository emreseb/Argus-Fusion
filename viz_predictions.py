#!/usr/bin/env python3
"""Render EO / IR / fused *prediction* boxes for one paired frame (3 images).

For a single synchronised EO+IR capture from the fusion test set this draws:
  1. docs/pred_eo.png           — EO model on the native EO frame        (green)
  2. docs/pred_ir.png           — IR model on the native IR frame        (red)
  3. docs/pred_fused_wbf.png    — fused output, box_mode=wbf, on IR      (cyan)
  4. docs/pred_fused_select.png — fused output, box_mode=select, on IR   (magenta)

The fused outputs come from the real pipeline (registration + fuzzy trust), so
they are exactly what the "Fuzzy (WBF)" / select rows in the results table score.
The two box modes share identical detections, so the only difference is how an
agreement pair's box is decided (averaged vs higher-trust sensor kept).

Run with the env that has ultralytics:
    ~/anaconda3/envs/YOLO-CUDA/bin/python viz_predictions.py
"""
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fusion"))

from pipeline import FusionPipeline, PipelineConfig  # noqa: E402

FT = "/home/emre/Desktop/NATO/DATASETv3/fusion_testset"
REG = os.path.join(FT, "eo2ir_affine.npy")
OUT = os.path.join(HERE, "docs")
CLASSES = ["drone", "bird"]


def read_label(path):
    if not os.path.exists(path):
        return []
    return [l.split() for l in open(path) if len(l.split()) >= 5]


def draw(img, boxes, color, labels):
    out = img.copy()
    for (x1, y1, x2, y2), txt in zip(boxes, labels):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 3)
        cv2.putText(out, txt, (p1[0], max(0, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def main():
    cfg = PipelineConfig(
        eo_model_path=os.path.join(HERE, "fusion/models/eo/EO-150-epoch.pt"),
        ir_model_path=os.path.join(HERE, "fusion/models/ir/IR-150-epoch.pt"),
        class_names=CLASSES,
        use_sahi=True,                 # a few frames only — fine for a viz
        use_registration=True,
        registration_matrix_path=REG,
        use_fuzzy_trust=True,
        box_mode="wbf",                # == "Fuzzy (WBF)" row in the results table
    )
    pipe = FusionPipeline(cfg)

    # --eo <PATH>  renders the grid for an arbitrary EO frame (IR pair = _0_->_1_).
    # --night uses a specific real night pair (N19) from the full dataset.
    # --dim picks the darkest fusion-testset scene (lands in TWILIGHT).
    # default = pinned day frame.
    EO_ARG = sys.argv[sys.argv.index("--eo") + 1] if "--eo" in sys.argv else None
    NIGHT = "--night" in sys.argv
    DIM = "--dim" in sys.argv
    SUFFIX = "_far" if EO_ARG else ("_night" if NIGHT else ("_dim" if DIM else ""))

    # extra GT label dir for frames outside fusion_testset (full dataset)
    EXTRA_LBL = "/home/emre/Desktop/NATO/DATASETv3/labels(removed empty txts)/obj_train_data"

    def load(stem):
        eo = cv2.imread(os.path.join(FT, "eo", stem + ".jpg"))
        ir = cv2.imread(os.path.join(FT, "ir", stem.replace("_0_", "_1_") + ".jpg"))
        return eo, ir

    if EO_ARG:
        eo_path = os.path.abspath(EO_ARG)
        stem = os.path.splitext(os.path.basename(eo_path))[0]
        ir_path = os.path.join(os.path.dirname(eo_path),
                               os.path.basename(eo_path).replace("_0_", "_1_"))
        eo_img, ir_img = cv2.imread(eo_path), cv2.imread(ir_path)
        if eo_img is None or ir_img is None:
            print(f"missing EO or IR image for {stem}")
            return 1
        pipe.reset()
        res = pipe.process_frame(eo_img, ir_img)
    elif NIGHT:
        NDIR = "/home/emre/Desktop/NATO/DATASETv3/new-category-full-dataset"
        stem = "1_100_0_N19_frame000001"                 # EO stem (sensor 0)
        eo_img = cv2.imread(os.path.join(NDIR, stem + ".jpg"))
        ir_img = cv2.imread(os.path.join(NDIR, stem.replace("_0_", "_1_") + ".jpg"))
        pipe.reset()
        res = pipe.process_frame(eo_img, ir_img)
    elif DIM:
        cands = []
        for eo_lbl in sorted(glob.glob(os.path.join(FT, "labels", "*.txt"))):
            b = os.path.basename(eo_lbl)
            ir_lbl = os.path.join(FT, "labels_ir", b.replace("_0_", "_1_"))
            if len(read_label(eo_lbl)) == 1 and len(read_label(ir_lbl)) == 1:
                cands.append(b[:-4])

        def meanv(stem):
            img = cv2.imread(os.path.join(FT, "eo", stem + ".jpg"))
            if img is None:
                return 1e9
            return float(cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2HSV)[:, :, 2].mean())

        order = sorted(cands, key=meanv)        # darkest first
        chosen = None
        for stem in order[:150]:                # the dimmer end of the set
            eo_img, ir_img = load(stem)
            if eo_img is None or ir_img is None:
                continue
            pipe.reset()
            res = pipe.process_frame(eo_img, ir_img)
            if [f for f in res.fused if set(f.support) == {"EO", "IR"}]:
                chosen = stem
                break
        if chosen is None:
            print("no dim agreement frame found")
            return 1
        stem = chosen
    else:
        stem = "1_001_0_M2_frame000009"
        eo_img, ir_img = load(stem)
        pipe.reset()
        res = pipe.process_frame(eo_img, ir_img)

    print(f"frame: {stem}  regime={res.regime.value}  meanV={res.mean_v:.1f}")
    print(f"  EO dets={len(res.eo_dets)}  IR dets={len(res.ir_dets)}  fused={len(res.fused)}")

    # ---- crop windows: one shared zoom for everything --------------------------
    # All three IR-frame panels use the SAME window (centred on the IR detection)
    # so WBF/select/IR are pixel-comparable. The EO window covers the matching
    # field of view (scaled by the EO/IR resolution ratio) so the drone shows at
    # the same apparent size. Everything is resized to TILE -> identical zoom.
    TILE = 640
    IR_HALF = 160                      # IR-frame crop half-size (px) -> 320px window
    eh, ew = eo_img.shape[:2]
    ih, iw = ir_img.shape[:2]
    scale = ew / iw                    # EO is higher-res; match the FOV
    EO_HALF = int(IR_HALF * scale)

    def center(b):
        return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))

    def window(cx, cy, half, W, H):
        x0 = int(min(max(cx - half, 0), W - 2 * half))
        y0 = int(min(max(cy - half, 0), H - 2 * half))
        return x0, y0, 2 * half, 2 * half

    # Drone ground-truth centre (testset labels, else full-dataset labels) so all
    # panels stay centred on the target even when a detector misses the far drone.
    A = np.load(REG)                                     # EO(scaled to IR grid) -> IR
    eo_gt = read_label(os.path.join(FT, "labels", stem + ".txt")) \
        or read_label(os.path.join(EXTRA_LBL, stem + ".txt"))
    gt_eo = (float(eo_gt[0][1]) * ew, float(eo_gt[0][2]) * eh) if eo_gt else None

    def eo_to_ir(pt):                                     # forward affine, EO px -> IR px
        d = A[:, :2] @ np.array([pt[0] * iw / ew, pt[1] * ih / eh]) + A[:, 2]
        return float(d[0]), float(d[1])

    if res.ir_dets:
        ir_cx, ir_cy = center(res.ir_dets[0].bbox_xyxy)
    elif gt_eo:
        ir_cx, ir_cy = eo_to_ir(gt_eo)
    else:
        ir_cx, ir_cy = iw / 2, ih / 2
    ir_win = window(ir_cx, ir_cy, IR_HALF, iw, ih)

    def crop_draw(img, win, boxes, labels, color):
        x0, y0, w, h = win
        crop = img[y0:y0 + h, x0:x0 + w].copy()
        s = TILE / float(w)
        crop = cv2.resize(crop, (TILE, TILE))
        for (bx1, by1, bx2, by2), txt in zip(boxes, labels):
            p1 = (int((bx1 - x0) * s), int((by1 - y0) * s))
            p2 = (int((bx2 - x0) * s), int((by2 - y0) * s))
            cv2.rectangle(crop, p1, p2, color, 2)
            cv2.putText(crop, txt, (p1[0], max(14, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return crop

    # 1) EO predictions, native EO frame, EO-FOV crop.
    # Run EO on the SAME preprocessed input the pipeline uses (e.g. gamma in
    # twilight) so the panel matches what actually fed fusion; draw on the raw
    # frame. Centre the crop on the drone's EO ground-truth location.
    from pipeline.preprocess import enhance
    p = pipe.params[res.regime]
    eo_in = enhance(eo_img, p.eo_preprocess) if p.eo_preprocess else eo_img
    eo_native = pipe.eo_detector.run(eo_in, conf=p.conf_thresh_EO)
    if gt_eo:                                            # GT centre
        eo_cx, eo_cy = gt_eo
    elif res.ir_dets:                                    # no GT: map IR det back to EO
        Minv = np.linalg.inv(A[:, :2])
        irc = np.array(center(res.ir_dets[0].bbox_xyxy))
        src = Minv @ (irc - A[:, 2])                     # EO centre in IR grid
        eo_cx, eo_cy = src[0] * ew / iw, src[1] * eh / ih
    elif eo_native:
        eo_cx, eo_cy = center(eo_native[0].bbox_xyxy)
    else:
        eo_cx, eo_cy = ew / 2, eh / 2
    eo_win = window(eo_cx, eo_cy, EO_HALF, ew, eh)
    eo_crop = crop_draw(eo_img, eo_win,
                        [d.bbox_xyxy for d in eo_native],
                        [f"{CLASSES[d.class_id]} {d.conf:.2f}" for d in eo_native],
                        (0, 200, 0))
    cv2.imwrite(os.path.join(OUT, f"pred_eo{SUFFIX}.png"), eo_crop)

    # 2) IR predictions, native IR frame, shared crop
    ir_crop = crop_draw(ir_img, ir_win,
                        [d.bbox_xyxy for d in res.ir_dets],
                        [f"{CLASSES[d.class_id]} {d.conf:.2f}" for d in res.ir_dets],
                        (0, 0, 255))
    cv2.imwrite(os.path.join(OUT, f"pred_ir{SUFFIX}.png"), ir_crop)

    # 3) fused output, both box modes, shared crop (re-fuse same dets, no inference)
    def render_fused(box_mode, color, fname):
        pipe.cfg.box_mode = box_mode
        r = pipe.process_detections(res.eo_dets, res.ir_dets, res.regime)
        crop = crop_draw(ir_img, ir_win,
                         [f.bbox_xyxy for f in r.fused],
                         [f"{'+'.join(f.support)} {CLASSES[f.class_id]} {f.conf:.2f}" for f in r.fused],
                         color)
        cv2.imwrite(os.path.join(OUT, fname), crop)

    render_fused("wbf", (255, 255, 0), f"pred_fused_wbf{SUFFIX}.png")    # cyan

    # 4) grid: EO | IR on top, fused (WBF) centred underneath
    def titled(fn, title):
        t = cv2.imread(os.path.join(OUT, fn))
        bar = np.zeros((34, TILE, 3), np.uint8)
        cv2.putText(bar, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, t])

    top = np.hstack([titled(f"pred_eo{SUFFIX}.png", "EO model (native EO)"),
                     titled(f"pred_ir{SUFFIX}.png", "IR model (native IR)")])
    fused = titled(f"pred_fused_wbf{SUFFIX}.png", "Fused (WBF)")
    pad = (top.shape[1] - fused.shape[1]) // 2            # centre the fused panel
    bottom = np.hstack([np.zeros((fused.shape[0], pad, 3), np.uint8), fused,
                        np.zeros((fused.shape[0], top.shape[1] - fused.shape[1] - pad, 3), np.uint8)])
    grid = np.vstack([top, bottom])
    cv2.imwrite(os.path.join(OUT, f"pred_grid{SUFFIX}.png"), grid)

    print(f"saved -> docs/pred_{{eo,ir,fused_wbf}}{SUFFIX}.png + docs/pred_grid{SUFFIX}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
