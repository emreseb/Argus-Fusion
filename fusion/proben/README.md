# ProbEn — Probabilistic Ensembling (late detection fusion)

Implementation of **Chen et al., "Multimodal Object Detection via Probabilistic
Ensembling", ECCV 2022** — a separate, drop-in alternative to the plan's
noisy-OR `decision_logic`, kept here so you can A/B it against the main pipeline.

## The idea

Given detections of the same object from K modalities, assumed conditionally
independent given the category, the fused posterior over classes is the
prior-corrected **product** of the per-modality class distributions:

```
p(y | d_1..d_K) ∝ p(y)^(1-K) · Π_k p(y | d_k)
```

A detector that emits a single foreground score `s` for class `c` is treated as
the categorical distribution `[… s at c …, (1-s) at background]`. With a uniform
prior this is the normalised elementwise product, e.g. two sensors agreeing at
0.8 fuse to `0.8·0.8 / (0.8·0.8 + 0.2·0.2) = 0.94`. Boxes are fused by
score-weighted averaging.

## How it differs from the plan's fusion

| | ProbEn | plan (noisy-OR) |
|---|---|---|
| score rule | Bayesian product of experts | noisy-OR + agreement bonus |
| agreement (0.8, 0.8) | **0.94** (rewards harder) | 0.77 |
| lighting/regime weighting | none | EO/IR weighted by day/night |
| lonely single sensor | keeps raw score | discounted by `lonely_penalty` |
| class disagreement | confidence collapses (background wins) | both kept as singles |

Run `python fusion/proben/demo_proben.py` to see all of these side by side
(numpy only, no models needed).

## Usage

```python
from proben.proben import proben_two, ProbEnConfig
fused = proben_two(eo_dets, ir_dets, ProbEnConfig(num_classes=2, assoc_iou=0.5))
# or N modalities:
from proben.proben import proben_fuse
fused = proben_fuse([eo_dets, ir_dets, radar_dets], ["EO", "IR", "RADAR"], cfg)
```

It consumes the pipeline's `Detection` and returns `FusedDetection`, so the eval
harness scores it as just another mode (`--modes proben`).

## Simplifications vs. the paper

- top-1 score → `{class, background}` distribution (the paper can use the full
  softmax when the detector exposes per-class logits).
- per-modality NMS is assumed already applied (YOLO output is).
- defaults to a uniform prior; pass `ProbEnConfig(prior=...)` for an empirical one.
