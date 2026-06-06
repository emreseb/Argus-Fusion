# EO/IR Detection Fusion Pipeline

Implementation of [`docs/fusion-pipeline-plan.md`](../docs/fusion-pipeline-plan.md):
brightness-aware **late (detection-level) fusion** of an EO and an IR detector
into one decided box + fused confidence per frame, then a Kalman tracker that
confirms targets over time.

> The loose alignment scripts already in this `fusion/` folder
> (`SIFT-align.py`, `auto-align-ecc.py`, …) are **not** part of this pipeline —
> the plan assumes co-boresighted sensors and skips spatial registration. They
> stay available for the day the hardware needs software co-registration.

## Layout

```
fusion/
├── pipeline/               # the package (importable; numpy-only core)
│   ├── schema.py           # Detection / FusedDetection            (plan §4)
│   ├── params.py           # Regime enum + per-regime table        (plan §6)
│   ├── regime.py           # EO brightness → DAY/TWILIGHT/NIGHT    (plan §2.2)
│   ├── preprocess.py       # EO gamma / CLAHE low-light boost      (plan §2.3)
│   ├── sync.py             # temporal pairing + resize, no warp    (plan §2.1)
│   ├── inference.py        # SAHI/YOLO Detector.run → [Detection]  (plan §2.3)
│   ├── fusion.py           # associate / WBF / noisy-OR / decision (plan §3)
│   ├── tracking.py         # constant-velocity Kalman + manager    (plan §5)
│   ├── pipeline.py         # FusionPipeline orchestrator           (plan §7)
│   ├── config.py           # PipelineConfig (all knobs)
│   └── viz.py              # debug overlay drawing
├── run_fusion.py           # CLI: run over two video files / frame folders
├── demo_synthetic.py       # numpy-only demo + self-test (no models needed)
└── requirements.txt
```

## Quick start

**1. Verify the logic with no ML deps** (needs only numpy):

```bash
python fusion/demo_synthetic.py
```

This drives a synthetic drone through day→twilight→night, printing the fused
confidence/support and the Kalman track each frame, and asserts the math of
plan §3 and §5. Good first check that everything is wired correctly.

**2. Run on real footage** (needs `pip install -r fusion/requirements.txt`):

```bash
python fusion/run_fusion.py \
    --eo eo.mp4 --ir ir.mp4 \
    --eo-model best.pt --ir-model bestmodels/best13.pt \
    --show                       # or --save out.mp4, or --print for headless
```

EO and IR inputs may each be a **video file** or a **directory of frames**;
frames are paired by index (sensors assumed frame-synced). Device (CUDA/MPS/CPU)
is auto-detected; force it with `--device`. Use `--no-sahi` for plain YOLO when
targets are large.

## Using it as a library

```python
from pipeline import FusionPipeline, PipelineConfig, Detection, Regime

cfg  = PipelineConfig(eo_model_path="best.pt", ir_model_path="best.pt")
pipe = FusionPipeline(cfg)

# real frames (BGR numpy arrays):
result = pipe.process_frame(eo_img, ir_img, frame_ts=t)
print(result.regime, [f.conf for f in result.fused], [trk.id for trk in result.tracks])

# or feed detections you already have (no models, numpy only):
result = pipe.process_detections(
    [Detection("EO", [x1,y1,x2,y2], 0.9)],
    [Detection("IR", [x1,y1,x2,y2], 0.8)],
    regime=Regime.NIGHT,          # or a regime name, or a raw mean-V float
)
```

`FrameResult` carries the `regime`, `mean_v`, raw `eo_dets`/`ir_dets`, the
`fused` detections, and the confirmed `tracks` (each with `bbox`, `center`,
`velocity`, stable `id`).

## How fusion decides (plan §3)

1. **Associate** EO↔IR boxes by IoU (greedy, highest first).
2. **Fuse box** — agreement pairs use Weighted Box Fusion (confidence × modality
   weight); singles keep their own box.
3. **Fuse confidence** — noisy-OR of the modality-weighted confidences, then an
   **agreement bonus** for confirmed pairs and a **lonely penalty** for
   unconfirmed singles, finally thresholded.

All weights/thresholds are per **regime** (DAY / TWILIGHT / NIGHT), chosen by an
EMA-smoothed brightness check on the EO stream. EO leans heavy by day, IR by
night — so an EO-only box at night is dropped while an IR-only box survives
(visible in the demo output).

## Tuning

- Edit thresholds in `pipeline/params.py`, or pass a JSON override via
  `--params file.json` / `PipelineConfig(params_path=...)`. Partial files are
  fine — unspecified fields inherit the defaults per regime.
- Brightness cuts: `--t-low` / `--t-high` (defaults 90 / 150 on mean V).
- Tracking gates: `track_min_hits`, `track_max_age`, `track_iou_gate` in config.
- Optional trajectory feedback (plan §5) — boost detections where a track is
  predicted, discount ones from nowhere — via `--feedback` /
  `PipelineConfig(use_track_feedback=True)`.

## Scaling beyond two sensors

The math already generalises (plan §8): `noisy_or(*probs)` takes any number of
weighted confidences, and WBF/association are pairwise building blocks. Adding a
third sensor means adding its weight column and clustering its boxes in.
