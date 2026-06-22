# EO/IR Drone Detection & Late-Fusion Pipeline

A research codebase for **detecting drones in dual-sensor (electro-optical + infrared) video** and
**fusing the two streams into a single, more reliable decision**. It pairs a modular fusion pipeline
with the full supporting toolchain: dataset building, annotation review, training, inference, and
evaluation — all in YOLO format.

The headline component is `fusion/pipeline/` — an end-to-end EO/IR fusion engine that handles
mis-aligned cameras, changing light, detection-level fusion, and Kalman tracking. The rest of the
repo is the scaffolding that produced and validated it.

> **Heads-up before you clone:** the dataset side of this repo is built around a **strict filename
> naming convention**. If your files don't follow it, the pairing, statistics, and frame-extraction
> tools won't work for you out of the box. See [The naming convention](#the-naming-convention-read-this)
> below — this is the single biggest portability limitation.

---

## What it does

| Capability | Where |
|---|---|
| **Late (detection-level) fusion** of EO + IR — IoU association → Weighted Box Fusion / ProbEn → noisy-OR confidence | `fusion/pipeline/fusion.py`, `fusion/pipeline/proben.py` |
| **Self-calibrating EO→IR registration** — recovers the affine transform automatically from detections (no checkerboard, no manual points), with RANSAC | `fusion/pipeline/registration.py` |
| **Brightness-regime adaptation** — DAY / TWILIGHT / NIGHT switch that retunes thresholds and per-sensor trust | `fusion/pipeline/params.py`, `fusion/pipeline/regime.py` |
| **Fuzzy sensor-trust engine** — smooth EO/IR weighting that replaces the hard regime table | `fusion/pipeline/fuzzy_trust.py` |
| **Kalman trajectory tracking** with greedy IoU matching, used as a prior for feedback | `fusion/pipeline/tracking.py` |
| **EO low-light preprocessing** — gamma / CLAHE enhancement | `fusion/pipeline/preprocess.py` |
| **Detection / evaluation metrics** — numpy-only precision/recall/F1/AP, plus EO-vs-IR comparisons | `fusion/eval/`, `model-eval/` |
| **Dataset tooling** — dual-sensor frame extraction, EO/IR pairing, train/test split, stats & plots | `frames/`, `dataset/`, `CVAT-Stats/` |
| **Annotation review** — a Flask web tool for flipping/fixing labels | `bit-flipper/` |
| **Live & viewer tools** — Streamlit viewers and SAHI tiled inference | `live/`, `view_all.py` |

---

## How the fusion pipeline works

```
       EO video ─┐                         ┌─► EO detector (YOLO / SAHI) ─┐
                 ├─ sync (timestamp pair) ─►│                             ├─► decision logic ─► Kalman tracks
       IR video ─┘                         └─► IR detector (YOLO / SAHI) ─┘   (associate → WBF/
                         │                                                     noisy-OR → threshold)
              EO brightness check                          ▲
              → DAY/TWILIGHT/NIGHT regime ─────────────────┘
                 (or fuzzy trust weighting)
```

Stage by stage (`fusion/pipeline/pipeline.py` orchestrates all of it):

1. **Sync & size** (`sync.py`) — temporally pairs EO and IR frames by timestamp and puts them in a
   shared pixel frame (resize only, no warping).
2. **Brightness regime** (`regime.py`, `params.py`) — the EO frame's mean brightness selects a
   lighting regime (DAY/TWILIGHT/NIGHT), which swaps in a full parameter set: per-sensor trust,
   confidence cuts, association IoU, agreement bonus, lonely-detection penalties, and the final
   decision threshold. Every value is overridable from JSON without touching code.
3. **Detection** (`inference.py`) — one uniform `Detector` API wraps a YOLO model per sensor, with
   optional **SAHI** sliced inference for small/distant targets.
4. **Registration** (`registration.py`) — the plan *assumes* the cameras are co-boresighted, but real
   hardware rarely is (different position **and** field of view → IoU is 0 → nothing fuses). This
   module solves the chicken-and-egg problem: on frames with exactly one confident EO box and one IR
   box the pairing is unambiguous, so it records correspondences, accumulates enough spatial spread,
   then fits and **locks** an affine transform via RANSAC.
5. **Decision logic** (`fusion.py`) — associates EO/IR boxes by IoU, fuses agreeing boxes (Weighted
   Box Fusion / ProbEn), combines confidences with noisy-OR, rewards independent confirmation,
   penalizes lonely single-sensor detections, and keeps detections above the regime's threshold.
6. **Tracking** (`tracking.py`) — a Kalman tracker maintains confirmed trajectories and produces
   one-step predictions that can feed back as priors.

There are **two entry points**: `process_frame()` (the real path: raw frames in, runs the detectors)
and `process_detections()` (model-free, numpy-only — for the synthetic demo, tests, or replaying
logged detections).

---

## Repository map

| Path | What's there |
|---|---|
| `fusion/pipeline/` | The fusion engine (config, sync, regime, inference, registration, fusion, tracking, fuzzy trust, preprocess, viz) |
| `fusion/eval/` | numpy-only detection metrics (precision/recall/F1/AP) |
| `fusion/run_fusion.py` | CLI driver for the whole pipeline |
| `fusion/demo_synthetic.py` | Self-contained synthetic day→night scenario + unit checks |
| `fusion/*.py` | Standalone registration experiments (SIFT, ECC, edge/MI alignment, homography) |
| `frames/` | Dual-sensor video → synced frame extraction |
| `dataset/` | Pairing, train/test split, dataset cleaning & audit scripts |
| `CVAT-Stats/` | Dataset statistics and condition/sensor breakdown plots |
| `bit-flipper/` | Flask web tool for reviewing and fixing labels |
| `trainers/` | YOLO training scripts (CPU / GPU / Mac) |
| `live/` | Inference tools, incl. SAHI tiled inference and a web viewer |
| `model-eval/` | Model evaluation scripts and one-shot eval |
| `yaml-files/` | Dataset / training configs |
| `view_all.py` | Streamlit viewer for images + labels |
| `docs/` | Design plan, EO/IR registration explainer, report drafts (PDF) |

---

## Quickstart

### Run the fusion pipeline

```bash
cd fusion
python run_fusion.py \
  --eo path/to/eo_video.mp4 \
  --ir path/to/ir_video.mp4 \
  --eo-model weights/eo.pt \
  --ir-model weights/ir.pt \
  --register \           # auto-fit the EO→IR affine transform
  --show                 # display annotated output
```

Useful flags: `--no-sahi` (plain YOLO instead of slicing), `--feedback` (trajectory priors),
`--reg-matrix file.npy` / `--save-reg file.npy` (load/save a locked registration),
`--t-low/--t-high` (brightness thresholds), `--params params.json` (per-regime overrides),
`--save out.mp4`, `--max-frames N`. Run `python run_fusion.py -h` for the full list.

No data handy? Try the dependency-light synthetic demo:

```bash
python fusion/demo_synthetic.py   # simulates a drone across day→twilight→night, asserts the core math
```

### Run the dataset / viewer tools

Most tools read paths from environment variables rather than hardcoded paths:

- `DATASET_ROOT` — dataset root (expects `images/` and `labels/` underneath)
- `MODEL_PATH` — YOLO model for inference/eval
- `SAHI_DEVICE` — `cpu` (default), `mps`, or `cuda`

```bash
export DATASET_ROOT="/path/to/dataset"
python -m streamlit run view_all.py
```

Dataset images/labels and model weights are **not** in git — point the tools at a local copy.

Install dependencies from the relevant `requirements.txt` (`fusion/requirements.txt` for the
pipeline). Core stack: `ultralytics` (YOLO11), `sahi`, OpenCV, NumPy.

---

## The naming convention (read this)

⚠️ **The dataset tools are useless on arbitrary filenames.** Frame extraction, EO/IR pairing, and
all the statistics assume a specific underscore-delimited filename scheme. If your data doesn't
follow it, those tools will silently skip your files or pair them wrong.

A filename encodes the recording's metadata in leading tokens:

```
   1 _ 110 _ 0 _ E19 . mp4
   │    │    │    └── Experiment ID — the "Common ID" (first non-numeric token, e.g. E19, ERF2, R5)
   │    │    └────── Sensor:  0 = EO, 1 = IR
   │    │
   │    └─────────── LDB three-digit condition code:
   │                   L (digit 0): Light        0 = Bright,  1 = Low
   │                   D (digit 1): Distance      0 = Close,   1 = Far
   │                   B (digit 2): Background     0 = Clear,   1 = Cluttered
   └──────────────── Class: 1 = Drone   (other classes are skipped by the stats tools)
```

Why it matters:

- **EO/IR pairing** keys on the **Common ID** (the `E`-number). `frame-exractor-v5.py` extracts it
  with `get_common_id()` and uses it to match an EO video to its IR twin and to name output folders
  (`EO/E19/…`, `IR/E19/…`). No matching Common ID → no pair → nothing to fuse.
- **Statistics** (`CVAT-Stats/`) parse the `LDB` code and sensor digit to break results down by
  light / distance / background / sensor. Files that don't parse are dropped.
- The default experiment prefix is configurable (`exp_type` in the extractor, e.g. `"E"` / `"ERF"`),
  but the **positional, underscore-delimited structure is hardcoded**. There is no generic config
  for a different scheme — you'd need to adapt the parsing functions.

**If you want to reuse these tools, either rename your data to match the scheme above, or edit
`extract_prefix_tokens()` / `get_common_id()` to your own convention.** The fusion pipeline itself
(`fusion/pipeline/`) does **not** depend on the naming convention — you feed it videos/frames and
models directly — so it is the portable part.

---

## Limitations & honest caveats

- **Naming convention lock-in** — as above; the biggest barrier to reuse of the dataset tooling.
- **Research code, not a product** — expect rough edges, experiment scripts, and assumptions baked
  in. Several alignment approaches (SIFT, ECC, edge, mutual-information, homography) are exploratory
  alternatives, not all production paths.
- **No data or weights in git** — `.pt`, images, and labels are gitignored. You bring your own.
- **Registration assumes a fixed rig** — one affine lock holds for a deployment; re-fit if the
  mount/zoom changes. It needs frames with a single unambiguous EO/IR target to bootstrap.
- **Parameters are starting points** — the per-regime table is transcribed from the design plan and
  meant to be tuned on real day *and* night footage.
- **Paths via environment variables** — set `DATASET_ROOT` / `MODEL_PATH` / `SAHI_DEVICE` or the
  tools won't find anything.

---

## What this repo gives you

- A **complete, modular reference** for EO/IR detection-level fusion you can read end to end — every
  stage is its own small module with a clear contract.
- A genuinely useful trick: **self-calibrating registration from detections alone**, solving the
  "need a transform to associate, need associations to fit a transform" deadlock without calibration
  targets.
- Two ways to combine sensors — a **discrete brightness-regime table** and a **smooth fuzzy-trust
  engine** — so you can compare hard vs. soft weighting.
- A **dependency-light path** (`process_detections()` + the synthetic demo) to experiment with the
  fusion math using only NumPy, no GPU or weights required.
- The surrounding **dataset/annotation/training/eval toolchain** that took raw dual-sensor video to a
  trained, evaluated model.

---

## License

Released under the [MIT License](LICENSE) © 2025-2026 Sebahattin Saral — free to use, modify, and
distribute; just keep the copyright notice.

---

<sub>🧭 This repository's architecture was mapped with
[graphify](https://github.com/sponsors/safishamsi) — 510 nodes / 914 edges across 60 communities,
which is how this README was assembled from the code and design docs.</sub>
