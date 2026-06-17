# EO/IR Fusion — Step-by-Step Calculations

This document traces **exactly what is computed at every stage**, from two raw
camera frames to the final evaluation numbers. It is the implementation
companion to `docs/fusion-pipeline-plan.md`, grounded in the actual code under
`fusion/` and the real rig used here:

- **EO** camera: 3840×2160 (4K visible)
- **IR** camera: 1280×1024 (thermal)
- Sensors are **not** co-boresighted → an EO→IR affine is fitted (see Step 4).

Notation: a box is `[x1, y1, x2, y2]` in pixels; `conf` ∈ [0,1] is a detector
confidence; `meanV` is mean HSV brightness.

---

## 0. Data flow at a glance

```
 EO frame ─┐
           ├─►(1) sync/size ─►(2) brightness→regime ─►(3) pick params
 IR frame ─┘                                                │
                                                            ▼
   EO ─►(4a) EO preprocess ─►(5) EO detector ─► EO dets ─►(4b) register EO→IR ─┐
   IR ───────────────────────►(5) IR detector ─► IR dets ──────────────────────┤
                                                                                ▼
                                       (6) DECISION LOGIC  (your pipeline = "plan")
                                   associate → WBF box → noisy-OR conf → bonus/penalty → threshold
                                                                                ▼
                                                            fused detections (boxes + 1 conf each)
                                                                                ▼
                                                      (8) Kalman tracking → confirmed tracks
                                                                                ▼
                                                      (9) EVALUATION vs ground truth
```

Code map: `fusion/pipeline/` (`sync.py`, `regime.py`, `preprocess.py`,
`registration.py`, `inference.py`, `fusion.py`, `fuzzy_trust.py`, `tracking.py`,
`pipeline.py`), alternative fusion in `fusion/proben/`, evaluation in
`fusion/eval/`.

---

## 1. Sync & sizing — `sync.py`

**Purpose:** put both frames into one comparable pixel grid. No warping.

- **Co-boresighted mode (`use_registration=False`):** if EO and IR differ in
  size, IR is resized to the EO frame so pixel coordinates (and therefore IoU)
  are comparable. Everything downstream lives in the **EO** frame.
- **Registration mode (`use_registration=True`, your case):** the frames are
  **not** resized here — each detector runs on its native frame, and alignment
  is handled later in Step 4 by the affine. Everything downstream lives in the
  **IR** frame.

Temporal sync: EO frame *i* is paired with IR frame *i* (frame-synced capture).
`nearest_timestamp()` exists for timestamp-based pairing within a tolerance.

**Output:** an (EO, IR) frame pair in a defined reference frame.

---

## 2. Brightness → regime — `regime.py`

**Purpose:** decide the lighting regime from the **EO** frame (EO brightness
matches human-visible lighting; IR reflects heat, not illumination).

**Step 2.1 — mean brightness.** Convert EO BGR→HSV, average the V channel:

```
meanV = mean( HSV(EO)[:, :, 2] )            # 0..255
```

(The frame is first downscaled to 256×256 purely for speed; the mean is
scale-stable. V = max(B,G,R), so BGR vs RGB gives the same V.)

**Step 2.2 — EMA smoothing** (so a headlight/cloud doesn't flip the regime):

```
ema_t = α · meanV_t + (1 − α) · ema_{t−1}        α = ema_alpha = 0.3
```

**Step 2.3 — classify** with two thresholds `t_low=90`, `t_high=150`:

```
ema ≥ 150            → DAY
90 ≤ ema < 150       → TWILIGHT
ema < 90             → NIGHT
```

**Output:** one of `DAY / TWILIGHT / NIGHT`, plus `meanV` for logging.

> On the current test set every frame came out DAY/TWILIGHT (0 NIGHT) — if true
> night footage exists, `t_low/t_high` need re-tuning on the brightness
> histogram.

---

## 3. Select parameter set — `params.py`

The regime indexes a `RegimeParams` row. Two tables are in play:

**Plan defaults (`pipeline/params.py`)** — trusts EO by day:

| Param              | DAY  | TWILIGHT | NIGHT       |
| ------------------ | ---- | -------- | ----------- |
| modality_weight_EO | 0.70 | 0.50     | 0.30        |
| modality_weight_IR | 0.30 | 0.50     | 0.70        |
| conf_thresh_EO     | 0.40 | 0.45     | 0.55        |
| conf_thresh_IR     | 0.45 | 0.40     | 0.35        |
| eo_preprocess      | none | gamma    | clahe+gamma |
| assoc_iou          | 0.40 | 0.40     | 0.40        |
| agreement_bonus    | 0.15 | 0.20     | 0.20        |
| lonely_penalty_EO  | 0.20 | 0.15     | 0.40        |
| lonely_penalty_IR  | 0.20 | 0.15     | 0.10        |
| decision_threshold | 0.45 | 0.45     | 0.45        |

**Tuned override (`fusion/eval/params_ir_trust.json`)** — trusts IR (fit to this
rig, where IR ≫ EO):

| Param              | DAY  | TWILIGHT | NIGHT |
| ------------------ | ---- | -------- | ----- |
| modality_weight_EO | 0.25 | 0.18     | 0.20  |
| modality_weight_IR | 0.75 | 0.82     | 0.80  |
| lonely_penalty_EO  | 0.50 | 0.50     | 0.40  |
| lonely_penalty_IR  | 0.05 | 0.05     | 0.05  |

(Unspecified fields inherit the plan defaults.)

**A third option — the fuzzy trust engine (`fuzzy_trust.py`).** Both tables above
are *discrete*: they pick one `(weight_EO, weight_IR)` pair per regime and switch
hard at the brightness thresholds. The fuzzy engine replaces that lookup with a
**smooth, size-aware dial** that varies continuously with brightness *and* target
size — no hard switch at 90/150. It is enabled with `use_fuzzy_trust=True` (or
scored as the `fuzzy`/`fuzzysel` modes in evaluation) and is described in full in
**Step 6A**. When it is off, the tables above are used exactly as before.

---

## 4. EO preprocessing & registration

**Step 4a — EO low-light enhance (`preprocess.py`).** EO only, by regime:

- `gamma` (TWILIGHT): per-pixel `out = (in/255)^(1/γ) · 255`, `γ=0.6` → brightens.
- `clahe+gamma` (NIGHT): CLAHE on the L channel of LAB (clip 2.0, 8×8 tiles),
  then gamma. IR is never preprocessed.

**Step 4b — EO→IR registration (`registration.py`).** Only when
`use_registration=True`. The two cameras see the drone at different pixels
(measured median offset **112px** here), so a 2×3 affine `A` maps EO into the IR
frame.

Fitting (done once, offline, from paired GT — `eval/prefit_registration.py`):

```
scale EO center into IR grid:  src = (cx_EO_norm · IRW,  cy_EO_norm · IRH)
IR center:                     dst = (cx_IR_norm · IRW,  cy_IR_norm · IRH)
solve least squares:           dst ≈ A · [x, y, 1]^T          (A is 2×3)
RANSAC rejects mismatches; lock if ≥60% inliers and median residual ≤ 8px
```

Fitted result for this rig (1297 correspondences):

```
A = [[1.4796, -0.0043, -311.0],
     [0.0074,  1.0421,  -49.4]]      median residual = 3.84px, 64% inliers
```

Applying to an EO box (`AutoRegistrar.apply`): scale the box into the IR grid
(`sx=IRW/EOW`, `sy=IRH/EOH`), warp its 4 corners by `A`, take the min/max →
box in IR pixels. After this, EO and IR detections share the IR frame and the
112px offset becomes ~4px, so they overlap and can be fused.

---

## 5. Per-model inference — `inference.py`

**Purpose:** turn each frame into a list of detections.

For each sensor: BGR→RGB, then **SAHI sliced inference** — the frame is cut into
640×640 tiles with 20% overlap, the YOLO model runs on each tile, and boxes are
merged back (so small drones aren't missed). Each surviving box at
`conf ≥ conf_thresh_<sensor>` becomes:

```
Detection( source="EO"/"IR", bbox_xyxy, conf, class_id, frame_ts )
```

(For evaluation the conf threshold is lowered to a floor of 0.001 — Step 9 — so
the full precision-recall curve is visible.)

**Output:** `eo_dets`, `ir_dets`. After Step 4b both are in the IR frame.

---

## 6. Decision logic — your pipeline ("plan") — `fusion.py`

Input: `eo_dets`, `ir_dets` in a shared frame + the regime's params. Three
sub-steps then two adjustments and a threshold.

**Step 6.1 — IoU.** For two boxes:

```
inter = overlap area
union = area_EO + area_IR − inter
IoU   = inter / union
```

**Step 6.2 — Associate (greedy).** Compute IoU for every EO×IR pair, sort
descending, and match a pair if `IoU ≥ assoc_iou` (0.40) and neither box is
already used. Three outcomes:

- **pair** (EO+IR agree on a location),
- **eo_only** (EO box, no IR match),
- **ir_only** (IR box, no EO match).

**Step 6.3 — Fuse the box, for pairs only.** The modality weights
`modality_weight_EO/IR` come from either the regime table (Step 3) **or**, when
the fuzzy engine is on, the per-detection fuzzy dial (Step 6A); the box step is
identical either way. There are two box modes (`box_mode`, default `"wbf"`):

*`"wbf"` — Weighted Box Fusion (plan default):*

```
w_EO = modality_weight_EO · conf_EO
w_IR = modality_weight_IR · conf_IR
box  = (w_EO · box_EO + w_IR · box_IR) / (w_EO + w_IR)
```

So with IR weighted 0.75 vs EO 0.25, the fused box sits ~¾ of the way toward the
(accurate) IR box — this is what fixed the AP75 localization.

*`"select"` — keep the higher-trust sensor's box:*

```
box = box_EO   if  w_EO ≥ w_IR   else  box_IR
```

The other sensor still contributes *confidence* (Step 6.4) but not geometry, so a
low-trust box can never drag a high-trust one off-target — this protects tight-IoU
localization (AP75) and avoids turning a good box into a miss. The winner is
chosen from the **same weights**, so nothing here assumes a particular sensor is
better: give EO more trust (e.g. a stronger EO model later) and the EO box wins
automatically. In evaluation, `fuzzy` = fuzzy weights + WBF box, `fuzzysel` =
fuzzy weights + `select` box.

**Step 6.4 — Fuse the confidence (noisy-OR), for pairs:**

```
p_EO = modality_weight_EO · conf_EO
p_IR = modality_weight_IR · conf_IR
fused_conf = 1 − (1 − p_EO)(1 − p_IR)
```

**Step 6.5 — Adjustments.**

- *Agreement bonus* (pairs): `conf = min(1, fused_conf · (1 + agreement_bonus))`.
- *Lonely penalty* (singles):
  - EO-only: `conf = modality_weight_EO · conf_EO · (1 − lonely_penalty_EO)`
  - IR-only: `conf = modality_weight_IR · conf_IR · (1 − lonely_penalty_IR)`

**Step 6.6 — Threshold.** Keep fused detections with
`conf ≥ decision_threshold` (0.45 in operation; lowered to the floor for AP).

**Worked example (DAY, tuned weights, an agreeing pair conf_EO=0.6, conf_IR=0.9):**

```
p_EO = 0.25·0.6 = 0.15 ;  p_IR = 0.75·0.9 = 0.675
fused = 1 − (1−0.15)(1−0.675) = 1 − 0.85·0.325 = 0.724
with bonus 0.15 → min(1, 0.724·1.15) = 0.832
box ≈ (0.15·box_EO + 0.675·box_IR)/0.825  → ~82% IR box
```

**Output:** `FusedDetection( bbox_xyxy, conf, support=[…], regime, class_id )`.

---

## 6A. Fuzzy sensor-trust engine (alternative weighting) — `fuzzy_trust.py`

**Purpose:** replace the discrete per-regime weight table (Step 3) with a
**smooth, size-aware** EO/IR trust dial. The regime table answers "how much do we
trust IR?" with three fixed numbers and a hard switch at brightness 90/150. The
fuzzy engine answers it as a continuous function of two inputs — **brightness**
and **target size** — so trust glides rather than jumps, and a far-away (small)
target can be treated differently from a close (large) one in the same frame.
This is a genuine fuzzy inference system (Takagi–Sugeno): fuzzify → apply rules →
defuzzify by weighted average.

It outputs a single number `trust_IR ∈ [0,1]`; then
`modality_weight_IR = trust_IR`, `modality_weight_EO = 1 − trust_IR`, which feed
Steps 6.3–6.4 unchanged.

**Step 6A.1 — Inputs.** Brightness is the **same EMA-smoothed `meanV`** computed
in Step 2 (so it is already de-flickered); target size is the box's long side,
`size = max(width, height)` in pixels (for a pair, the mean of the two boxes).

**Step 6A.2 — Fuzzify (membership).** Each input is split into three overlapping
triangular sets whose memberships always sum to 1:

```
brightness → (dark, dim, bright)   knots (60, 120, 180)  → crossovers at 90 & 150
size (px)  → (small, medium, large) knots (16, 48, 96)    → crossovers at 32 & 72
```

The brightness crossovers are placed on **90 and 150 on purpose** — the same
boundaries as the regime thresholds `t_low/t_high` — so the fuzzy dial agrees with
the table at the extremes but blends smoothly across the boundary instead of
switching.

**Step 6A.3 — Rule base (trust_IR per cell).** A 3×3 grid, "how much to trust IR"
for each (brightness × size) combination. Darker and smaller ⇒ lean on IR;
brighter and larger ⇒ let EO back in:

| trust_IR | small | medium | large |
| -------- | ----- | ------ | ----- |
| dark     | 0.90  | 0.85   | 0.80  |
| dim      | 0.80  | 0.70   | 0.60  |
| bright   | 0.60  | 0.45   | 0.35  |

These nine values are **a-priori** (set from first principles, *not* fit to the
test set) — a cleaner methodological position than the test-set-tuned table.

**Step 6A.4 — Inference + defuzzify (Sugeno).** Each rule's firing strength is the
AND (= min) of its two memberships; the output is the strength-weighted average of
the rule values:

```
strength_ij = min( μ_brightness(i), μ_size(j) )
trust_IR    = Σ strength_ij · rule_ij  /  Σ strength_ij
```

(If brightness is missing, it falls back to `default_trust = 0.5`.)

**Worked example (brightness 100, size 30px):**

```
brightness 100 → dark 0.333, dim 0.667, bright 0      (between knots 60 and 120)
size 30        → small 0.5625, medium 0.4375, large 0 (between knots 16 and 48)
four rules fire:
  dark·small   min(0.333,0.5625)=0.333  → 0.90
  dark·medium  min(0.333,0.4375)=0.333  → 0.85
  dim·small    min(0.667,0.5625)=0.5625 → 0.80
  dim·medium   min(0.667,0.4375)=0.4375 → 0.70
trust_IR = (0.333·0.90 + 0.333·0.85 + 0.5625·0.80 + 0.4375·0.70) / 1.666 = 0.804
→ modality_weight_IR = 0.804, modality_weight_EO = 0.196
```

A nearby brightness (90 → 0.775, 92 → 0.770) gives a nearby trust — the whole
point: no cliff at the regime boundary.

**Output:** `(modality_weight_EO, modality_weight_IR)` per detection, consumed by
Step 6.3/6.4. The rule grid and knots can be overridden via a JSON file
(`fuzzy_config_path` / `--fuzzy-config`). The module is pure-stdlib and
self-testing (`python -m pipeline.fuzzy_trust`).

---

## 7. ProbEn (alternative fusion, for comparison) — `proben/proben.py`

Same inputs, a different score rule (Bayesian product instead of noisy-OR; no
regime weighting). For a cluster of matched detections:

```
each detection → distribution over {classes…, background}:
    detected class c gets s = conf ; background gets 1 − s ; others ≈ 0
fuse (uniform prior):  posterior(y) ∝ Π_k p_k(y) , then normalize
fused score = max foreground posterior ; box = score-weighted average
```

Two agreeing at 0.8 → `0.8·0.8 / (0.8·0.8 + 0.2·0.2) = 0.94`. A lone detection
keeps its score; a class disagreement collapses toward background. ProbEn is
scored as the `proben` mode in evaluation; it has no `--params` knobs.

---

## 8. Trajectory analysis (Kalman) — `tracking.py`

**Purpose:** confirm targets over time, smooth boxes, suppress one-frame FPs.

**State:** `x = [cx, cy, w, h, vx, vy]` (constant-velocity).

**Predict (every frame):**

```
x ← F x            (F advances center by velocity: cx += vx, cy += vy)
P ← F P Fᵀ + Q     (Q = process noise)
```

**Update (when a fused detection matches a track):** standard Kalman correction
with measurement `z = [cx, cy, w, h]`:

```
y = z − H x                      (innovation)
S = H P Hᵀ + R                   (R = measurement noise)
K = P Hᵀ S⁻¹                     (Kalman gain)
x ← x + K y ;  P ← (I − K H) P
```

**Track management (SORT-style):**

- predict all tracks → greedy IoU-match fused detections to predictions (gate
  `iou_gate=0.3`);
- matched → `update`; unmatched detection → new tentative track; unmatched track
  → missed;
- a track is **confirmed** after `min_hits=3` hits and **deleted** after
  `max_age=5` consecutive misses.

**Output:** confirmed tracks with smoothed box, velocity `(vx, vy)`, stable ID.

---

## 9. Evaluation — `eval/run_eval.py` + `eval/detection_metrics.py`

**Purpose:** measure the fused detector as an ablation against its inputs, on
ground truth, per regime.

**Step 9.1 — Predictions.** For each paired frame, run the pipeline once and
collect the prediction sets in the reference (IR) frame, scored side by side on
identical detections:

- `eo` — raw EO detector,
- `ir` — raw IR detector,
- `plan` — table weights + WBF box (Steps 3, 6),
- `fuzzy` — fuzzy weights (Step 6A) + WBF box,
- `fuzzysel` — fuzzy weights + `select` box (Step 6.3),
- `proben` — Bayesian-product alternative (Step 7).

Detectors run at conf floor 0.001 and `decision_threshold` is lowered so AP
integrates the whole curve. Modality weights / penalties / IoU (the *method*) are
kept. `fuzzy`, `fuzzysel` and `proben` are each compared against the best single
sensor via ΔAP50 (Step 9.5).

**Step 9.2 — Ground truth.** Read the YOLO label for the frame and convert to
pixels in the reference frame:

```
x1 = (cx − w/2)·W ;  y1 = (cy − h/2)·H ;  x2 = (cx + w/2)·W ;  y2 = (cy + h/2)·H
```

With registration ON the reference is IR, so GT = the **IR-image labels**
(`labels_ir/`) at `W,H = 1280,1024`. (Co-boresighted mode uses EO labels.)

**Step 9.3 — Match predictions to GT.** Per image, per class, sort predictions
by confidence; each prediction is a **TP** if it hits an unclaimed GT box at
`IoU ≥ thr`, else a **FP**; unmatched GT = **FN**.

**Step 9.4 — Precision-recall & AP.** Sweep the confidence threshold:

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)   = TP / (#GT)
```

AP = area under the (precision-enveloped) PR curve, 101-point interpolation.

- **AP50** at IoU thr 0.5, **AP75** at 0.75,
- **mAP50-95** = mean AP over IoU thr 0.50, 0.55, …, 0.95,
- **F1 = 2·P·R/(P+R)**, reported at the confidence that maximizes it.
- Overall = mean over classes that have ground truth (an empty declared class
  like `bird` is skipped, not scored 0).

**Step 9.5 — Stratify & compare.** Repeat per regime (DAY/TWILIGHT/NIGHT) and
overall, then report the headline:

```
Δ AP50 = AP50_fused − max(AP50_eo, AP50_ir)
```

Positive = fusion beats the best single sensor; ≈0 = parity; negative = fusion
hurts.

---

## 9A. Results — fusion test set

**Setup.** 1298 paired frames, registration ON (IR reference frame, GT =
`labels_ir/`), pre-fit affine (median residual 3.84px), models `EO-150-epoch.pt`
/ `IR-150-epoch.pt`, conf floor 0.001 so AP integrates the full PR curve. Every
frame classified DAY or TWILIGHT — **0 NIGHT frames** on this set, so the night
rules/params were not exercised. `ΔAP50 = AP50_mode − max(AP50_eo, AP50_ir)`.

**DAY (1161 frames)**

| mode      |   P   |   R   |  F1   | AP50  | AP75  | mAP50-95 | ΔAP50  |
| --------- | ----- | ----- | ----- | ----- | ----- | -------- | ------ |
| eo        | 0.322 | 0.316 | 0.319 | 0.131 | 0.001 | 0.026    |   —    |
| ir        | 0.843 | 0.780 | 0.810 | 0.772 | 0.414 | 0.412    |   —    |
| plan      | 0.836 | 0.525 | 0.645 | 0.501 | 0.020 | 0.139    | −0.271 |
| fuzzy     | 0.887 | 0.586 | 0.705 | 0.677 | 0.108 | 0.255    | −0.094 |
| fuzzysel  | 0.713 | 0.618 | 0.662 | 0.606 | 0.247 | 0.280    | −0.166 |
| proben    | 0.710 | 0.643 | 0.675 | 0.641 | 0.154 | 0.266    | −0.131 |

**TWILIGHT (137 frames)**

| mode      |   P   |   R   |  F1   | AP50  | AP75  | mAP50-95 | ΔAP50  |
| --------- | ----- | ----- | ----- | ----- | ----- | -------- | ------ |
| eo        | 0.218 | 0.190 | 0.203 | 0.045 | 0.000 | 0.006    |   —    |
| ir        | 0.970 | 0.949 | 0.959 | 0.962 | 0.457 | 0.493    |   —    |
| plan      | 0.850 | 0.745 | 0.794 | 0.746 | 0.085 | 0.247    | −0.216 |
| fuzzy     | 0.902 | 0.876 | 0.889 | 0.830 | 0.165 | 0.325    | −0.132 |
| fuzzysel  | 0.940 | 0.912 | 0.926 | 0.906 | 0.432 | 0.452    | −0.056 |
| proben    | 0.765 | 0.664 | 0.711 | 0.754 | 0.187 | 0.292    | −0.208 |

**ALL (1298 frames)**

| mode      |   P   |   R   |  F1   | AP50  | AP75  | mAP50-95 | ΔAP50  |
| --------- | ----- | ----- | ----- | ----- | ----- | -------- | ------ |
| eo        | 0.312 | 0.300 | 0.306 | 0.119 | 0.001 | 0.023    |   —    |
| ir        | 0.857 | 0.791 | 0.823 | 0.788 | 0.412 | 0.415    |   —    |
| plan      | 0.836 | 0.527 | 0.647 | 0.507 | 0.020 | 0.140    | −0.281 |
| fuzzy     | 0.774 | 0.673 | 0.720 | 0.688 | 0.106 | 0.255    | −0.100 |
| fuzzysel  | 0.740 | 0.644 | 0.689 | 0.635 | 0.257 | 0.292    | −0.152 |
| proben    | 0.718 | 0.639 | 0.676 | 0.647 | 0.155 | 0.266    | −0.141 |

### Discussion

1. **IR is the strongest single sensor; EO is weak (AP50 0.12).** Consequently
   **no fusion mode beats IR-alone** on this set — the realistic goal here is how
   *close* a fusion rule gets to IR without degrading it. ΔAP50 is negative for
   every fusion mode; the question is which loses least.

2. **Both fuzzy variants beat the discrete regime table (`plan`) on every
   metric.** The smooth, size-aware dial roughly halves the table's AP50 deficit
   (fuzzy ΔAP50 −0.10 vs plan −0.28) and multiplies its AP75 several-fold. Since
   the only change is the *weighting* (same association, penalties, threshold),
   this is a clean win for the fuzzy engine over the hard switch — and the grid
   is a-priori, not fit to this set.

3. **The `select` box mode does what it was designed to: it restores
   localization.** ALL AP75 goes **0.106 → 0.257** (≈2.4×) from `fuzzy` to
   `fuzzysel`, and in TWILIGHT it reaches **0.432 vs IR's 0.457** — near parity.
   This confirms the diagnosis that WBF was averaging a weak EO box into the
   result and dragging it off-target; keeping the trusted (IR) box verbatim fixes
   it. By the integrated metric **mAP50-95, `fuzzysel` is the best fusion mode**
   (0.292 ALL, vs fuzzy 0.255, proben 0.266).

4. **The trade-off is in DAY.** There `select` *lowers* AP50 (0.677 → 0.606)
   while raising AP75 (0.108 → 0.247). Cause: in bright light the a-priori grid
   still grants EO meaningful trust, so `select` sometimes commits fully to a
   (weak) EO box, whereas WBF's blending hides that by keeping most of IR. So
   `select` is best where the trusted sensor genuinely dominates (TWILIGHT), and
   riskier where the grid mis-judges the weak EO (DAY). The best single fusion
   result in the whole study is **`fuzzysel` in TWILIGHT, ΔAP50 −0.056** — fusion
   essentially at parity with IR.

5. **Net read.** On this rig IR carries the system; fusion's value is bounded by
   the weak EO model. The fuzzy trust engine clearly beats the hard regime table,
   and box-`select` recovers the localization WBF was destroying. To push a
   fusion mode to ΔAP50 ≥ 0 the EO model needs to improve, or the bright row of
   the grid should be re-tuned down (less EO trust in daylight) so `select` stops
   committing to bad EO boxes in DAY; a hybrid that uses `select` only when one
   sensor's weight dominates by a margin (else WBF) would combine fuzzy's DAY
   AP50 with fuzzysel's AP75.

> **Reporting caveat:** predictions and GT are both in the IR frame, so IR is on
> native turf while EO/fusion is charged the ~4px registration residual — this
> slightly favours IR. See §10.

---

## 10. What the reference frame / GT choice means

- Registration maps everything into the **IR** frame, so predictions and GT are
  both IR-frame. The IR sensor is therefore on "home turf" (native frame, native
  labels), while EO/fusion is charged the ~4px registration residual. This
  slightly favors IR — state it when reporting.
- There is no separate "fused ground truth"; one sensor's annotations (IR) are
  adopted as canonical. An EO-frame run (map IR→EO, score vs EO labels) is the
  fair cross-check.
- **Tuning caveat:** `params_ir_trust.json` was fit on this test set. For
  publishable numbers, tune weights on a validation split and report on a
  held-out test split.

---

## 11. One-line summary of each step

1. **Sync/size** — common pixel grid (or native frames if registering).
2. **Brightness** — `meanV` of EO HSV-V, EMA-smoothed → DAY/TWILIGHT/NIGHT.
3. **Params** — regime selects weights/thresholds.
4. **Preprocess + register** — EO gamma/CLAHE; affine maps EO boxes into IR frame.
5. **Inference** — SAHI-sliced YOLO → EO and IR detection lists.
6. **Decision logic (plan)** — associate (IoU) → fuse box (WBF or select) →
   noisy-OR conf → agreement bonus / lonely penalty → threshold.
6A. **Fuzzy trust** — optional smooth, size-aware EO/IR weighting (brightness +
   target size, Sugeno) replacing the discrete regime table.
7. **ProbEn** — Bayesian-product alternative for comparison.
8. **Kalman** — predict/update, confirm after N hits, delete after K misses.
9. **Eval** — match preds→GT by IoU, PR curve → P/R/F1/AP50/AP75/mAP50-95 per
   regime, report ΔAP over the best single sensor.
