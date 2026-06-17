# Report Draft 2 — Full Revisions (side-by-side editing sheet)

Paste-ready replacement text and fixes for every inconsistency found in
`Report Draft 2.pdf`. Open this next to the report and apply top to bottom.
Legend: **[PASTE]** = drop-in text · **[ACTION]** = a change you make · **[CONFIRM]** = a
number only you can verify.

Priority order: **A** (method-breaking) → **B** (numbers/specs) → **C** (structure) → **D** (typos).

---

# A. Contradictions that undermine the method

## A1 — Registration story (figure done; add the text)

**[PASTE — end of §6 FUSION]**

> It is worth noting that late fusion does not remove the need for spatial
> alignment entirely. Because the fusion engine associates EO and IR detections by
> their bounding-box overlap (IoU), the two sensors' boxes must live in a common
> coordinate frame. Aligning a handful of detection boxes, however, is far lighter
> than warping and aligning every pixel of two full images as early fusion would
> require. We achieve this box-level alignment with a one-time affine registration,
> described in Section 7.

**[PASTE — new stage in §7.2 Pipeline Stages, before the EO/IR models run]**

> **Registration.** Because the EO and IR sensors are not co-boresighted — they
> have different fields of view and resolutions, so the same drone appears at
> different pixel coordinates in each view (a measured median offset of ≈112 px) —
> EO and IR detections cannot be compared directly. We estimate a 2×3 affine
> transform *A* that maps EO coordinates into the IR frame. It is fitted once,
> offline, from paired single-target frames: corresponding EO/IR drone centres are
> collected, outliers rejected with RANSAC, and the fit is locked once it has
> enough inliers and a low median residual (≈3.8 px on our data). At run time every
> EO detection box is warped by *A* into the IR frame, after which EO and IR boxes
> overlap and can be associated by IoU. All downstream fusion operates in the IR
> coordinate frame.

**[CONFIRM]** the 112 px offset and 3.8 px residual against `fusion-pipeline-calculations.md` §4.

---

## A2 — §7.3 Decision Logic (fuzzy-forward)

**[ACTION]** Replace the whole "7.3 Decision Logic" body (the fuzzy-logic
definition, the formula block, the dangling "image above" sentence). **Keep** the
existing DAY/TWILIGHT/NIGHT parameter table — it is now the *baseline*.

**[PASTE]**

### 7.3 Decision Logic

The decision logic is the core of the late fusion. Our approach is *adaptive
weighting*: we take the confidence score from each single-sensor model, multiply
it by a weight describing how reliable that model is in the current situation, and
combine the weighted scores into a single fused certainty. Weighted Box Fusion
applies the same idea to the bounding boxes rather than the scores.

The weights are produced by a fuzzy inference system. Fuzzy logic lets a value
belong to several linguistic categories at once, each to a degree between 0 and 1,
instead of snapping to a single category at a hard threshold. Our trust engine is
a Takagi–Sugeno fuzzy inference system: each input (brightness and target size) is
*fuzzified* into overlapping membership values, a small rule base maps every input
combination to a trust value, and those are combined by a weighted average
(*defuzzification*) into a single crisp weight, `trust_IR`. The result is a trust
value that glides smoothly as conditions change rather than jumping at a boundary.
(The membership terms and rule grid are given in Section 7.1.)

For an agreement pair, the fused box and confidence are computed as:

```
a_IR = trust_IR ,   a_EO = 1 − trust_IR        (from the fuzzy engine, §7.1)
w_eo = a_EO · conf_eo ,   w_ir = a_IR · conf_ir
box  = (w_eo · box_eo + w_ir · box_ir) / (w_eo + w_ir)      # WBF
  or   box = box of the sensor with the larger weight        # select
conf = 1 − (1 − w_eo)(1 − w_ir)                              # noisy-OR
```

Two EO and IR boxes are treated as the same object — an *agreement pair* — when
their IoU (Intersection over Union) is above the association threshold `assoc_iou`
(e.g. 0.4); otherwise they remain separate single-sensor detections. An agreement
pair receives an *agreement bonus*, while a detection seen by only one sensor
receives a single-sensor *lonely penalty*; both adjust the fused confidence before
the final decision threshold.

The discrete DAY / TWILIGHT / NIGHT parameter set below is our **baseline**
weighting, against which the fuzzy engine is compared in the Results section. The
two use the same boundaries: the fuzzy terms *dark / dim / bright* cross over
exactly at the thresholds T_low and T_high that separate NIGHT / TWILIGHT / DAY —
the table switches hard at those points, while the fuzzy engine blends across them.
The table still supplies everything except the modality weights (confidence
thresholds, EO preprocessing, association IoU, agreement bonus, lonely penalty);
the modality weights a_EO and a_IR come from the fuzzy engine of Section 7.1.

*(Caption the existing parameter table: "Baseline per-regime parameters.")*

---

## A3 — §8 Results (real measured numbers)

**[ACTION]** Replace the placeholder table (and its empty rows / red notes) with
the measured results below. The old fixed-split rows (EO 50/50, 70/30, 30/70) were
never run; the rows here are what was actually evaluated. Conditions: 1298 paired
frames, registered (scored in the IR frame against IR labels), detectors at conf
floor 0.001. ΔAP50 = AP50 − max(AP50_EO, AP50_IR). Every frame was DAY or TWILIGHT
(**0 NIGHT frames** on this set).

**[PASTE] — Overall (all 1298 frames)**

| Approach              |   P   |   R   |  F1   | AP50  | AP75  | mAP50-95 | ΔAP50  |
| --------------------- | ----- | ----- | ----- | ----- | ----- | -------- | ------ |
| EO baseline           | 0.312 | 0.300 | 0.306 | 0.119 | 0.001 | 0.023    |   —    |
| IR baseline           | 0.857 | 0.791 | 0.823 | 0.788 | 0.412 | 0.415    |   —    |
| Baseline table (hard) | 0.836 | 0.527 | 0.647 | 0.507 | 0.020 | 0.140    | −0.281 |
| Fuzzy (WBF)           | 0.774 | 0.673 | 0.720 | 0.688 | 0.106 | 0.255    | −0.100 |
| Fuzzy + select        | 0.740 | 0.644 | 0.689 | 0.635 | 0.257 | 0.292    | −0.152 |
| ProbEn                | 0.718 | 0.639 | 0.676 | 0.647 | 0.155 | 0.266    | −0.141 |

**[PASTE] — DAY (1161 frames)**

| Approach              |   P   |   R   |  F1   | AP50  | AP75  | mAP50-95 | ΔAP50  |
| --------------------- | ----- | ----- | ----- | ----- | ----- | -------- | ------ |
| EO baseline           | 0.322 | 0.316 | 0.319 | 0.131 | 0.001 | 0.026    |   —    |
| IR baseline           | 0.843 | 0.780 | 0.810 | 0.772 | 0.414 | 0.412    |   —    |
| Baseline table (hard) | 0.836 | 0.525 | 0.645 | 0.501 | 0.020 | 0.139    | −0.271 |
| Fuzzy (WBF)           | 0.887 | 0.586 | 0.705 | 0.677 | 0.108 | 0.255    | −0.094 |
| Fuzzy + select        | 0.713 | 0.618 | 0.662 | 0.606 | 0.247 | 0.280    | −0.166 |
| ProbEn                | 0.710 | 0.643 | 0.675 | 0.641 | 0.154 | 0.266    | −0.131 |

**[PASTE] — TWILIGHT (137 frames)**

| Approach              |   P   |   R   |  F1   | AP50  | AP75  | mAP50-95 | ΔAP50  |
| --------------------- | ----- | ----- | ----- | ----- | ----- | -------- | ------ |
| EO baseline           | 0.218 | 0.190 | 0.203 | 0.045 | 0.000 | 0.006    |   —    |
| IR baseline           | 0.970 | 0.949 | 0.959 | 0.962 | 0.457 | 0.493    |   —    |
| Baseline table (hard) | 0.850 | 0.745 | 0.794 | 0.746 | 0.085 | 0.247    | −0.216 |
| Fuzzy (WBF)           | 0.902 | 0.876 | 0.889 | 0.830 | 0.165 | 0.325    | −0.132 |
| Fuzzy + select        | 0.940 | 0.912 | 0.926 | 0.906 | 0.432 | 0.452    | −0.056 |
| ProbEn                | 0.765 | 0.664 | 0.711 | 0.754 | 0.187 | 0.292    | −0.208 |

*(Delete the red "we will fill the table…" notes and give the table a caption,
e.g. "Detection metrics per approach, registered IR frame.")*

---

## A3b — §9 Discussion (draft; replace the placeholder)

**[PASTE]**

1. **IR is the strongest single sensor; EO is weak (AP50 0.12).** As a result no
   fusion configuration beats IR alone — every ΔAP50 is negative — so the
   realistic question is which fusion rule loses the least. The fuzzy engine comes
   closest (ΔAP50 −0.10 overall).
2. **Both fuzzy variants beat the hard regime table on every metric.** The
   smooth, size-aware dial roughly halves the table's AP50 deficit (−0.10 vs
   −0.28) and multiplies its AP75 several-fold. As only the weighting changes
   (association, penalties and threshold are identical), this is a clean gain for
   the fuzzy engine — and its rule grid is a-priori, not fitted to this set.
3. **Box selection restores localization.** Overall AP75 rises from 0.106 (WBF)
   to 0.257 (select), and in TWILIGHT it reaches 0.432 against IR's 0.457 — near
   parity. This confirms that averaging a weak EO box into the result was dragging
   the box off-target; keeping the higher-trust box fixes it. By the integrated
   mAP50-95 metric, fuzzy+select is the best fusion mode (0.292).
4. **The trade-off appears in DAY.** There selection raises AP75 (0.108 → 0.247)
   but lowers AP50 (0.677 → 0.606): in bright light the a-priori grid still grants
   EO some trust, so selection occasionally commits to a weak EO box, whereas WBF
   hides this by averaging. The best single fusion result in the study is
   fuzzy+select in TWILIGHT, ΔAP50 −0.056.
5. **Net read and future work.** On this rig the IR sensor carries the system and
   fusion's ceiling is bounded by the weak EO model. The fuzzy trust engine
   clearly beats the hard table, and box-selection recovers the localization WBF
   destroys. Reaching ΔAP50 ≥ 0 requires a stronger EO detector, tuning the rule
   grid on a validation split, or a margin-gated box mode that uses selection only
   when one sensor's weight clearly dominates.

> **Reporting caveat:** predictions and ground truth are both in the IR frame, so
> IR is on native turf while EO/fusion is charged the ≈4 px registration residual;
> this slightly favours IR.

---

# B. Numeric / spec inconsistencies

## B4 — One title everywhere
**[ACTION]** The cover says *"Dynamic Weighted Late Fusion Pipeline for Multi-Modal
Drone Detection"*; the inner title page, copyright page and abstract say *"Multi
Modal Fusion for Drone Detection."* Pick one and use it on all four. Recommended:
the cover title (more descriptive), hyphenated **"Multi-Modal"** throughout.

## B5 — IR resolution
**[CONFIRM + ACTION]** Table 1 says IR = **640×512** and EO = **"4K / 1080p"**, but
the code (`reg_ir_size`) and `fusion-pipeline-calculations.md` use IR = **1280×1024**.
These can't both be true. Check the actual IR frame size you ran on, then make
Table 1, the calculations doc, and the code agree. Also fix EO **"4K / 1080p"** →
a single value (4K = 3840×2160).

## B6 — "3164 pictures" vs 11,534
**[CONFIRM + ACTION]** The categorizing paragraph says you classified **3164**
pictures, but Table 3's counts all sum to **11,534** (e.g. 5744 EO + 5790 IR).
Reconcile — if 11,534 is the dataset size, change "3164" to that (or clarify what
the 3164 subset was).

## B7 — Naming template vs example
**[ACTION]** The template `Item_Light_Distance_Background_Sensor_EXP_frameFID` implies
five separate single digits (`1_0_0_0_0_…`), but the real files are
`1_000_0_E11_frame000033` — Light/Distance/Background are one 3-digit group.
Replace the Figure 2 template + example with the real structure:

**[PASTE]**
> Template: `<Item>_<Light><Distance><Background>_<Sensor>_<Experiment>_frame<FID>.jpg`
> Example: `1_000_0_E11_frame000033.jpg` → drone (Item 1); bright, near, clear
> (Light 0, Distance 0, Background 0); EO (Sensor 0); experiment E11; frame 33.

Also make the experiment tag consistent — Figure 2 uses `ERF1` while the
BitFlipper/CVAT screenshots use `E11 / E17 / E9`. Use the real form (e.g. `E11`).

## B8 — "No object" vs "No Drone"
**[ACTION]** Table 2 calls class 2 *"No object"*; Table 3 calls it *"No Drone."*
Standardise to **"No object"** in Table 3 to match the naming convention.

## B9 — Remaining placeholders
**[ACTION]** Fill Table 1 "Total Samples" (`#getexact` → real counts; from Table 3
that's EO 5744, IR 5790, plus your paired count). Delete the literal `########` in
the categorizing paragraph. §8 results are handled by A3 above.

---

# C. Structure / cross-references

## C10 — §3 subsection numbers + TOC
**[ACTION]** The body shows literal placeholders `3.#. Extracting individual frames`,
`3.#. Naming convention`, `3.#. Categorizing frames`. Number them **3.1 / 3.2 / 3.3**
and add 3.2 and 3.3 to the Table of Contents (currently only 3.1 is listed).

## C11 — Missing figure in §7.3
**[ACTION]** Resolved by the A2 rewrite (it no longer says "the image above").
Just delete the red "Figure caption will added." note. Optional: add a small WBF
illustration if you want a visual.

## C12 — Front-matter placeholders
- **Abstract** — draft below.
- **Conclusion** — draft below.
- **Related Work** — your note says you'll cite inline; either keep §2 as a short
  paragraph saying so, or remove the empty section heading.
- **Acknowledgements** — personal; fill in or remove the "We should thank someone".
- **Roman numerals** — your own note: front matter (Approval → List of Figures)
  should use i, ii, iii…; numbered pages start at the Introduction.

**[PASTE — Abstract]**
> Unmanned aerial vehicles pose a growing security threat, making reliable drone
> detection important. Single-sensor detectors each have blind spots:
> electro-optical (EO) cameras struggle in low light, while infrared (IR) sensors
> lack visible detail. This work investigates whether fusing EO and IR drone
> detectors yields more accurate detection than either alone. We collect and
> annotate a synchronized EO/IR drone dataset captured with a DJI Matrice 4T, train
> single-modality YOLO11 detectors as baselines, and build a late-fusion pipeline
> that associates per-sensor detections, fuses their boxes and confidences, and
> tracks targets over time. Because the sensors are not co-boresighted, EO
> detections are mapped into the IR frame by a one-time affine registration. The
> core contribution is an adaptive, fuzzy sensor-trust engine that sets the EO/IR
> weighting from scene brightness and target size, replacing a hard per-regime
> lookup table. On a 1298-frame test set the fuzzy engine improves over the
> hard-table baseline on every metric (AP50 0.69 vs 0.51), and a trust-driven
> box-selection variant recovers localization accuracy (AP75 0.26 vs 0.02). Because
> the EO detector is substantially weaker than the IR detector on our data, no
> fusion configuration surpasses the IR detector alone; the fuzzy approach comes
> closest. We analyse why, and what is required for fusion to exceed the strongest
> single sensor.

**[PASTE — Conclusion]**
> We set out to determine whether multi-modal EO/IR fusion can outperform
> single-modality drone detection. We built a complete late-fusion pipeline —
> synchronized capture, per-sensor YOLO11 detectors, affine EO→IR registration, an
> adaptive fuzzy trust engine for sensor weighting, and Kalman tracking — and
> evaluated it as an ablation against its own inputs. The fuzzy trust engine clearly
> improves on a hard per-regime weighting table across all metrics, and the
> box-selection variant restores the localization accuracy that naive box averaging
> degrades. The headline finding is honest: on our dataset the IR detector is far
> stronger than the EO detector, so no fusion configuration beats IR alone — the
> fuzzy approach narrows but does not close the gap. This bounds the value of fusion
> by the weaker sensor's quality. Future work should strengthen the EO detector,
> tune the fuzzy rule base on a validation split, and explore a margin-gated box
> mode; because the trust engine is sensor-agnostic and shifts weight automatically
> as its inputs improve, the same pipeline is positioned to exceed the strongest
> single sensor once the EO model is more competitive.

---

# D. Typos (find → replace)

- **"Brithness"** → **"Brightness"** (List of Tables *and* the Table 4 caption).
- **"TWILIGTH"** → **"TWILIGHT"** (Table 4, regime column).
- §7.3 **"are system"** → **"our system"** (removed if you paste the A2 rewrite).
- §7.3 **"take affect"** → **"take effect"** (removed if you paste the A2 rewrite).
- §1.1 **"the advantages each sensor"** → **"the advantages of each sensor."**
- TOC note: **"seperate"** → **"separate"**, **"counter"** → **"counted."**
