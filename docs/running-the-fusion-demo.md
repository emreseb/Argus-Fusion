# Running the EO/IR fusion demo

Step-by-step to reproduce the EO/IR late-fusion demo on the NATO drone footage:
two non-co-boresighted streams are detected independently, registered into a
shared (IR) coordinate frame with a fixed affine, fused into one decided box +
confidence per frame, and tracked with a Kalman filter.

For *why* registration is needed (the sensors differ in pointing **and** field of
view), see [`eo-ir-registration-explained.md`](eo-ir-registration-explained.md).
For the fusion math, see [`../fusion/README.md`](../fusion/README.md).

## 0. Prerequisites

Everything lives under `/home/emre/Desktop/NATO`:

| What | Path |
|------|------|
| Code / pipeline | `code/fusion/` |
| EO weights | `code/fusion/models/eo/EO-150-epoch.pt` |
| IR weights | `code/fusion/models/ir/IR-150-epoch.pt` |
| EO video | `vid-src/EO-stream.mp4` (1920×1080-ish, 775 frames) |
| IR video | `vid-src/IR-stream.mp4` (1280×1024) |
| Pre-fit EO→IR affine | `code/fusion/out/affine_eo2ir.npy` |
| Demo outputs | `code/fusion/out/` |

The models need `ultralytics` + `torch` (CUDA). They are **not** in the base
`python`; use the `YOLO-CUDA` conda env:

```bash
source /home/emre/anaconda3/etc/profile.d/conda.sh
conda activate YOLO-CUDA
# sanity check: should print ultralytics 8.3.x | torch 2.9.x+cu128 | cuda True
python -c "import ultralytics, torch; print(ultralytics.__version__, torch.__version__, torch.cuda.is_available())"
```

Run all commands below from `/home/emre/Desktop/NATO/code`.

## 1. The demo (load the pre-fit affine, fuse from frame 0)

This is the canonical demo. It loads the already-calibrated EO→IR affine, so
there is **no warm-up** — fusion fires on frame 0.

```bash
python fusion/run_fusion.py \
    --eo /home/emre/Desktop/NATO/vid-src/EO-stream.mp4 \
    --ir /home/emre/Desktop/NATO/vid-src/IR-stream.mp4 \
    --eo-model fusion/models/eo/EO-150-epoch.pt \
    --ir-model fusion/models/ir/IR-150-epoch.pt \
    --reg-matrix fusion/out/affine_eo2ir.npy \
    --save fusion/out/fused_loaded.mp4 \
    --print | tee fusion/out/loaded.log
```

- `--reg-matrix` loads the affine and turns registration on.
- `--save` writes the annotated MP4 (drawn in the IR frame, since that's the
  shared coordinate space).
- `--print` logs one line per frame; `tee` keeps a copy in `loaded.log`.
- Swap `--save ...` for `--show` to watch a live window (add `--step` to advance
  frame-by-frame: any key = next, `q` = quit). `--show` needs a display.
- Add `--max-frames 50` for a quick smoke test.

Expected per-frame log (fusing immediately, `reg[LOCKED]`):

```
[00000] DAY      meanV=173.1 EO=1 IR=1 | fused[EO+IR:0.64] | tracks[-]        reg[LOCKED n=1 r=nanpx]
[00002] DAY      meanV=173.1 EO=1 IR=1 | fused[EO+IR:0.61] | tracks[#1@(133,804)] reg[LOCKED n=3 r=nanpx]
...
Processed 775 frame pair(s).
```

Reading a line: `regime`, EO-stream brightness `meanV`, raw EO/IR detection
counts, the `fused` boxes as `support:confidence` (`EO+IR` = both sensors
agreed), confirmed Kalman `tracks` as `#id@(x,y)`, and registration status.

Output video: `fusion/out/fused_loaded.mp4`.

## 2. (Optional) Re-calibrate the affine from scratch

If you ever need to regenerate `affine_eo2ir.npy` (e.g. new footage), let the
pipeline self-calibrate: on frames with exactly one confident EO and one IR box
the EO→IR correspondence is unambiguous, so it accumulates points, fits a RANSAC
affine once the target has roamed enough, **locks**, then saves it.

```bash
python fusion/run_fusion.py \
    --eo /home/emre/Desktop/NATO/vid-src/EO-stream.mp4 \
    --ir /home/emre/Desktop/NATO/vid-src/IR-stream.mp4 \
    --eo-model fusion/models/eo/EO-150-epoch.pt \
    --ir-model fusion/models/ir/IR-150-epoch.pt \
    --register \
    --save-reg fusion/out/affine_eo2ir.npy \
    --save fusion/out/fused_autoreg.mp4 \
    --print | tee fusion/out/autoreg.log
```

- `--register` enables self-calibration (early frames show `reg[warmup ...]` and
  `fused[-]` until it locks).
- `--save-reg` writes the locked affine at the end (residual ~2.5 px over ~600
  pairs on this footage).

Once saved, deploy with §1 (`--reg-matrix`) for zero warm-up.

## 1b. Live interactive window (scrub back and forth)

The MP4 runs above are headless. For the **live window** (title
*"EO→IR registered fusion"*) — a single drawn frame you can step through and
**reverse** — use `fuse_registered.py`. It hardcodes the EO/IR video paths, the
`models/eo` + `models/ir` weights, and defaults the affine to
`fusion/out/affine_eo2ir.npy`, so it takes no path args.

It needs an X display, so prefix with `QT_QPA_PLATFORM=xcb DISPLAY=:0`. **Run it
in your own terminal**, not in the background — the OpenCV window only receives
key presses when it owns the keyboard focus of an interactive session. Launched
detached/non-interactively, `waitKey` returns immediately and it just auto-plays
through every frame and exits.

```bash
QT_QPA_PLATFORM=xcb DISPLAY=:0 python fusion/fuse_registered.py --show --step --print
```

Controls (with `--step`):

| Key | Action |
|-----|--------|
| `→` or `d` | next frame |
| `←` or `a` | previous frame (scrub back) |
| `q` / `Esc` | quit |

A yellow HUD at the bottom shows `frame i/N` and the keys. Stepping is
**reversible**: each frame is run through the models + tracker exactly once, at
the frontier, then cached — going back just redraws the cache, so it never
re-runs inference or corrupts the Kalman tracker. Drop `--step` to auto-play
forward. `fuse_registered.py` also takes `--save out.mp4`, `--conf`, and
`--max-frames` (headless when `--show` is omitted).

> **Why this window seems faster than the §1/§2 MP4 runs.** Two reasons, both
> real (the predictions are genuine YOLO outputs either way): (1) with `--step`
> it only processes a frame when you press a key, vs. all 775 back-to-back; and
> (2) `fuse_registered.py` calls `model.predict()` **once per frame**, while
> `run_fusion.py` defaults to `use_sahi=True` (SAHI slices each frame into tiles
> and runs the model on every tile — many inferences per frame). Pass
> `--no-sahi` to `run_fusion.py` to close most of that gap.

## 3. Logic check with no ML deps (optional)

To verify the fusion/tracking math without models or GPU (numpy only, base
python is fine):

```bash
python fusion/demo_synthetic.py
```

Drives a synthetic drone through day→twilight→night and asserts the noisy-OR
fusion and Kalman track behave as specified.

## Troubleshooting

- **`python: command not found` (but `pip` works)** — there is no bare `python`
  on the system PATH, only `python3`, and the system `pip` is for python 3.12
  (no ML deps). The `python` command appears only inside the env:
  `conda activate YOLO-CUDA`, then use `python` (not `python3`).
- **`ModuleNotFoundError: ultralytics` / `torch`** — you're on base/system
  python; `conda activate YOLO-CUDA` first.
- **`RequestsDependencyWarning: Unable to find ... chardet or charset_normalizer`**
  — harmless. It comes from `requests` (pulled in by ultralytics); nothing in the
  pipeline makes HTTP calls. Ignore it, or silence with
  `pip install charset_normalizer` inside the env.
- **`fused[-]` on every frame with `--register`** — the affine never locked
  (target too stationary to constrain scale/rotation). Use the pre-fit matrix
  (§1) or run on footage where the drone roams.
- **`Could not open video` / `No images found`** — check the `--eo`/`--ir`
  paths; each may be a video file or a directory of frames (paired by index).
- **No window with `--show`** — needs a display; on headless boxes use `--save`
  / `--print` instead.
- **Slow** — the default uses plain inference per frame on GPU; ~a few minutes
  for all 775 frames. Add `--no-sahi` if you enabled SAHI slicing and targets
  are large, or `--max-frames N` to cap the run.
