# Fusion evaluation

Evaluates the fusion pipeline the way fusion *should* be evaluated: as an
**ablation** against its own inputs, stratified by lighting regime. A fused
detector is only "good" if it beats the best single sensor — especially in the
conditions where one sensor is weak.

## What it reports

For each mode — `eo`, `ir`, `plan` (noisy-OR fusion), `proben` (Probabilistic
Ensembling) — it prints **precision / recall / F1 / AP50 / AP75 / mAP50-95**,
overall and per regime (DAY / TWILIGHT / NIGHT), plus the headline:

```
Δ AP50 (plan   − best single)
Δ AP50 (proben − best single)
```

That delta is the number that actually answers "does fusion help, and where".

## Inputs

Directories of frames; GT is per-image **YOLO txt** (`class cx cy w h`,
normalised), matched to the EO frame by basename and assumed to be in the shared
(EO) coordinate frame (co-boresighted sensors). EO and IR frames are paired by
sorted order.

```
testset/
├── eo/      *.jpg        (EO frames)
├── ir/      *.jpg        (IR frames, same count/order)
└── labels/  *.txt        (YOLO GT, basenames match eo/)
```

## Run

```bash
python fusion/eval/run_eval.py \
    --eo testset/eo --ir testset/ir --labels testset/labels \
    --eo-model best.pt --ir-model fusion/models/ir/IR-150-epoch.pt \
    --modes eo ir plan proben \
    --out fusion/eval/results.csv
```

## Notes

- **Detectors run at a low conf floor** (`--conf-floor`, default 0.001) so AP
  integrates the full precision-recall curve. The regime weights, penalties and
  association IoU that define each *fusion method* are kept intact — only the
  detector confidence cut and the fusion `decision_threshold` are lowered for
  measurement.
- Metrics are numpy-only (`detection_metrics.py`), COCO-style 101-point AP, mean
  over classes that have ground truth (an empty declared class is skipped, not
  scored 0). Operating-point P/R/F1 are reported at the max-F1 confidence.
- This is **detection-level** evaluation. Track-level MOT metrics (MOTA / IDF1 /
  HOTA via TrackEval) are a sensible next layer — not included here.
