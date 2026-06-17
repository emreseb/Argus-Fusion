#!/usr/bin/env python3
"""Evaluate the fusion pipeline as an ablation: EO-only vs IR-only vs fused.

The point of fusion is the *lift* over its own inputs, so this runs the same
paired, GT-labelled test set through several "modes" and prints a comparison
table — overall and stratified by lighting regime (DAY/TWILIGHT/NIGHT) — plus
the headline ΔAP = AP_fused − max(AP_EO, AP_IR).

Modes
-----
    eo      raw EO detector output
    ir      raw IR detector output
    plan    the plan's fusion (associate + WBF + noisy-OR), decision_logic
    proben  Probabilistic Ensembling (fusion/proben)

Inputs are **directories of frames** (GT is per-image YOLO txt). EO and IR
frames are paired by sorted order; GT is matched to the EO frame by basename and
is assumed to live in the shared (EO) coordinate frame.

Example
-------
    python fusion/eval/run_eval.py \
        --eo testset/eo --ir testset/ir --labels testset/labels \
        --eo-model best.pt --ir-model fusion/models/ir/IR-150-epoch.pt \
        --modes eo ir plan proben --out fusion/eval/results.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

# Make sibling packages importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
sys.path.insert(0, _FUSION)

from pipeline import FusionPipeline, PipelineConfig  # noqa: E402
from eval.detection_metrics import evaluate  # noqa: E402
from eval.yolo_gt import load_yolo_labels, label_path_for  # noqa: E402

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_REGIMES = ("DAY", "TWILIGHT", "NIGHT")


def list_images(d: str):
    files = []
    for e in _IMG_EXTS:
        files += glob.glob(os.path.join(d, "**", f"*{e}"), recursive=True)
    return sorted(files)


def dets_to_preds(dets):
    return [(d.bbox_xyxy, float(d.conf), int(d.class_id)) for d in dets]


def build_config(args) -> PipelineConfig:
    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    if args.eo_model:
        cfg.eo_model_path = args.eo_model
    if args.ir_model:
        cfg.ir_model_path = args.ir_model
    if args.device:
        cfg.device = args.device
    if args.no_sahi:
        cfg.use_sahi = False
    if args.params:
        cfg.params_path = args.params
    if args.register:
        cfg.use_registration = True
    if args.reg_matrix:
        cfg.use_registration = True
        cfg.registration_matrix_path = args.reg_matrix
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eo", required=True, help="EO frame directory")
    ap.add_argument("--ir", required=True, help="IR frame directory")
    ap.add_argument("--labels", required=True, help="YOLO GT directory (shared/EO frame)")
    ap.add_argument("--eo-model", help="EO weights (.pt)")
    ap.add_argument("--ir-model", help="IR weights (.pt)")
    ap.add_argument("--config", help="PipelineConfig JSON")
    ap.add_argument("--params", help="per-regime params JSON (plan §6 overrides; affects 'plan' mode)")
    ap.add_argument("--device", help="auto | cpu | cuda:0 | mps")
    ap.add_argument("--no-sahi", action="store_true")
    ap.add_argument("--register", action="store_true",
                    help="learn EO->IR affine (non-coboresighted sensors); GT must be IR-frame labels")
    ap.add_argument("--reg-matrix", help="load a pre-fit affine .npy instead of warming up")
    ap.add_argument("--modes", nargs="+",
                    default=["eo", "ir", "plan", "fuzzy", "fuzzysel", "proben"],
                    choices=["eo", "ir", "plan", "fuzzy", "fuzzysel", "proben"])
    ap.add_argument("--conf-floor", type=float, default=0.001,
                    help="run detectors/fusion at this low conf so AP integrates the full PR curve")
    ap.add_argument("--fuzzy-config", help="fuzzy-trust JSON (rule grid / knots); affects 'fuzzy' mode")
    ap.add_argument("--proben-iou", type=float, default=0.5, help="ProbEn cross-modality assoc IoU")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", help="write the metrics table to this CSV")
    args = ap.parse_args()

    cfg = build_config(args)
    # 'plan' must stay the table baseline; 'fuzzy' is scored separately below so
    # the two appear side-by-side on identical detections.
    cfg.use_fuzzy_trust = False
    pipe = FusionPipeline(cfg)

    # Lower thresholds so AP sees the whole curve (weights/penalties/IoU kept).
    for rp in pipe.params.values():
        rp.conf_thresh_EO = args.conf_floor
        rp.conf_thresh_IR = args.conf_floor
        rp.decision_threshold = args.conf_floor

    import cv2
    proben = None
    if "proben" in args.modes:
        from proben.proben import proben_two, ProbEnConfig
        proben = (proben_two, ProbEnConfig(num_classes=len(cfg.class_names),
                                           assoc_iou=args.proben_iou,
                                           min_score=args.conf_floor))

    fuzzy_engine = None
    fuzzy_decision = None
    if "fuzzy" in args.modes or "fuzzysel" in args.modes:
        from pipeline.fusion import decision_logic as fuzzy_decision
        from pipeline.fuzzy_trust import FuzzyTrust, load_fuzzy_config
        fz_cfg = load_fuzzy_config(args.fuzzy_config) if args.fuzzy_config else None
        fuzzy_engine = FuzzyTrust(fz_cfg)

    eo_files = list_images(args.eo)
    ir_files = list_images(args.ir)
    if not eo_files or not ir_files:
        print("ERROR: no frames found in --eo / --ir", file=sys.stderr)
        return 1
    n = min(len(eo_files), len(ir_files))
    if args.max_frames:
        n = min(n, args.max_frames)
    print(f"Evaluating {n} frame pairs | modes={args.modes}")

    class_ids = list(range(len(cfg.class_names)))
    # preds[mode][regime] -> list of per-image predictions ; gts[regime] aligned.
    buckets = list(_REGIMES) + ["ALL"]
    preds = {m: {b: [] for b in buckets} for m in args.modes}
    gts = {b: [] for b in buckets}

    for i in range(n):
        eo = cv2.imread(eo_files[i])
        ir = cv2.imread(ir_files[i])
        if eo is None or ir is None:
            print(f"WARN: unreadable pair at index {i}", file=sys.stderr)
            continue

        res = pipe.process_frame(eo, ir, frame_ts=float(i))
        regime = res.regime.value
        reg_status = res.extra.get("registration")
        # With registration everything is mapped into the IR frame, so score
        # against IR-frame GT; otherwise the shared frame is EO.
        if cfg.use_registration:
            h, w = ir.shape[:2]
            gt = load_yolo_labels(label_path_for(ir_files[i], args.labels), w, h)
        else:
            h, w = eo.shape[:2]
            gt = load_yolo_labels(label_path_for(eo_files[i], args.labels), w, h)

        mode_preds = {}
        if "eo" in args.modes:
            mode_preds["eo"] = dets_to_preds(res.eo_dets)
        if "ir" in args.modes:
            mode_preds["ir"] = dets_to_preds(res.ir_dets)
        if "plan" in args.modes:
            mode_preds["plan"] = dets_to_preds(res.fused)
        if "fuzzy" in args.modes or "fuzzysel" in args.modes:
            # Same associate/penalty/threshold logic as 'plan', but the EO/IR
            # weights come from the fuzzy trust dial (brightness + target size).
            # 'fuzzy' averages the pair box (WBF); 'fuzzysel' keeps the
            # higher-trust sensor's box and uses the other only for confidence.
            p = pipe.params[res.regime]
            for mode_name, bmode in (("fuzzy", "wbf"), ("fuzzysel", "select")):
                if mode_name not in args.modes:
                    continue
                fdets = fuzzy_decision(
                    res.eo_dets, res.ir_dets, p, res.regime,
                    require_same_class=cfg.require_same_class,
                    fuzzy=fuzzy_engine, brightness=res.smoothed_v,
                    box_mode=bmode,
                )
                mode_preds[mode_name] = dets_to_preds(fdets)
        if "proben" in args.modes:
            fn, pcfg = proben
            mode_preds["proben"] = dets_to_preds(fn(res.eo_dets, res.ir_dets, pcfg, regime))

        for b in (regime, "ALL"):
            gts[b].append(gt)
            for m in args.modes:
                preds[m][b].append(mode_preds[m])

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{n}")

    if cfg.use_registration:
        if reg_status and reg_status.get("locked"):
            print(f"\n[registration] LOCKED — {reg_status['n_pairs']} pairs, "
                  f"residual {reg_status['residual_px']:.1f}px (EO mapped into IR frame)")
        else:
            print("\n[registration] NEVER LOCKED — EO detections were NOT mapped into the "
                  "IR frame, so the 'eo' row is EO boxes scored against IR-frame GT (≈0). "
                  "Pre-fit an affine and pass --reg-matrix, or relax RegistrationConfig.")

    # ---- compute + print ----
    rows = []
    print("\n" + "=" * 78)
    for b in buckets:
        if not gts[b] or sum(len(g) for g in gts[b]) == 0:
            continue
        n_gt = sum(len(g) for g in gts[b])
        print(f"\n### {b}  ({len(gts[b])} frames, {n_gt} GT boxes)")
        print(f"  {'mode':<8}{'P':>8}{'R':>8}{'F1':>8}{'AP50':>9}{'AP75':>9}{'mAP50-95':>10}")
        ap50 = {}
        for m in args.modes:
            r = evaluate(preds[m][b], gts[b], class_ids, cfg.class_names)["overall"]
            ap50[m] = r["AP50"]
            print(f"  {m:<8}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}"
                  f"{r['AP50']:>9.3f}{r['AP75']:>9.3f}{r['mAP50-95']:>10.3f}")
            rows.append([b, m, r["precision"], r["recall"], r["f1"],
                         r["AP50"], r["AP75"], r["mAP50-95"], n_gt])
        # Lift of each fusion mode over the best single sensor.
        base = max(ap50.get("eo", 0.0), ap50.get("ir", 0.0))
        for fm in ("plan", "fuzzy", "fuzzysel", "proben"):
            if fm in ap50:
                print(f"    Δ AP50 ({fm} − best single) = {ap50[fm] - base:+.3f}")
    print("\n" + "=" * 78)

    if args.out:
        with open(args.out, "w", newline="") as fh:
            wri = csv.writer(fh)
            wri.writerow(["regime", "mode", "precision", "recall", "f1",
                          "AP50", "AP75", "mAP50-95", "n_gt"])
            wri.writerows(rows)
        print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
