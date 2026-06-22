# Graph Report - .  (2026-06-22)

## Corpus Check
- Corpus is ~33,387 words - fits in a single context window. You may not need a graph.

## Summary
- 510 nodes · 914 edges · 60 communities (51 shown, 9 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 158 edges (avg confidence: 0.56)
- Token cost: 82,438 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Fusion Config & Core Types|Fusion Config & Core Types]]
- [[_COMMUNITY_Frame Fusion Runtime|Frame Fusion Runtime]]
- [[_COMMUNITY_Registration & Decision Concepts|Registration & Decision Concepts]]
- [[_COMMUNITY_Affine Registration Fitting|Affine Registration Fitting]]
- [[_COMMUNITY_Kalman Tracking|Kalman Tracking]]
- [[_COMMUNITY_Detection Metrics & Eval|Detection Metrics & Eval]]
- [[_COMMUNITY_ProbEn Probabilistic Fusion|ProbEn Probabilistic Fusion]]
- [[_COMMUNITY_Dataset Reviewer (Bit-Flipper)|Dataset Reviewer (Bit-Flipper)]]
- [[_COMMUNITY_Fuzzy Trust Engine|Fuzzy Trust Engine]]
- [[_COMMUNITY_Dual-Sensor Frame Extraction|Dual-Sensor Frame Extraction]]
- [[_COMMUNITY_EO Low-Light Preprocessing|EO Low-Light Preprocessing]]
- [[_COMMUNITY_CVAT Stat Plotting|CVAT Stat Plotting]]
- [[_COMMUNITY_YOLOSAHI Detector Wrapper|YOLO/SAHI Detector Wrapper]]
- [[_COMMUNITY_SAHI Web Inference Viewer|SAHI Web Inference Viewer]]
- [[_COMMUNITY_Mutual-Information Alignment|Mutual-Information Alignment]]
- [[_COMMUNITY_EOIR Dataset Audit|EO/IR Dataset Audit]]
- [[_COMMUNITY_Synthetic Demo & Unit Checks|Synthetic Demo & Unit Checks]]
- [[_COMMUNITY_Frame Sync & Sizing|Frame Sync & Sizing]]
- [[_COMMUNITY_YOLO Box Viewer|YOLO Box Viewer]]
- [[_COMMUNITY_HSVTRGBT Image Preprocessing|HSVT/RGBT Image Preprocessing]]
- [[_COMMUNITY_Video Annotation & Relabel|Video Annotation & Relabel]]
- [[_COMMUNITY_Registration Visualization|Registration Visualization]]
- [[_COMMUNITY_Dataset Stats|Dataset Stats]]
- [[_COMMUNITY_Manual Point Selection|Manual Point Selection]]
- [[_COMMUNITY_Extraction Debug & Prep|Extraction Debug & Prep]]
- [[_COMMUNITY_EOIR Image Counting|EO/IR Image Counting]]
- [[_COMMUNITY_TrainTest Split|Train/Test Split]]
- [[_COMMUNITY_One-Shot Eval Metrics|One-Shot Eval Metrics]]
- [[_COMMUNITY_Side-by-Side View|Side-by-Side View]]
- [[_COMMUNITY_Video to Dataset Frames|Video to Dataset Frames]]
- [[_COMMUNITY_SAHI Auto-Labeler|SAHI Auto-Labeler]]
- [[_COMMUNITY_numpy dependency (fusion core)|numpy dependency (fusion core)]]
- [[_COMMUNITY_opencv-python dependency|opencv-python dependency]]
- [[_COMMUNITY_split-folders dependency|split-folders dependency]]
- [[_COMMUNITY_torch dependency|torch dependency]]

## God Nodes (most connected - your core abstractions)
1. `Detection` - 36 edges
2. `FusedDetection` - 33 edges
3. `FusionPipeline` - 28 edges
4. `Regime` - 25 edges
5. `FuzzyTrust` - 24 edges
6. `decision_logic()` - 21 edges
7. `FrameResult` - 20 edges
8. `AutoRegistrar` - 19 edges
9. `Track` - 18 edges
10. `Detector` - 17 edges

## Surprising Connections (you probably didn't know these)
- `SAHI per-model sliced inference` --references--> `ultralytics YOLO backend`  [INFERRED]
  docs/fusion-pipeline-plan.pdf → fusion/requirements.txt
- `YOLO11 single-modality baseline models` --references--> `ultralytics YOLO backend`  [INFERRED]
  docs/Report Draft 2.pdf → fusion/requirements.txt
- `SAHI per-model sliced inference` --references--> `sahi sliced inference`  [INFERRED]
  docs/fusion-pipeline-plan.pdf → fusion/requirements.txt
- `main()` --calls--> `FusionPipeline`  [INFERRED]
  viz_predictions.py → fusion/pipeline/pipeline.py
- `ndarray` --uses--> `Detection`  [INFERRED]
  fusion/pipeline/inference.py → fusion/pipeline/schema.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Fusion decision core (associate, fuse box, fuse confidence)** — fusion_pipeline_plan_association_iou, fusion_pipeline_plan_weighted_box_fusion, fusion_pipeline_plan_noisy_or, fusion_pipeline_plan_decision_logic [EXTRACTED 1.00]
- **EO/IR self-calibrating registration flow** — eo_ir_registration_explained_not_coboresighted, eo_ir_registration_explained_affine_transform, eo_ir_registration_explained_self_calibration, eo_ir_registration_explained_ransac [EXTRACTED 1.00]
- **Fusion approach evaluation comparison** — report_revisions_results_table, report_revisions_box_select, report_revisions_fuzzy_trust_engine, report_revisions_proben, report_revisions_ir_strongest [EXTRACTED 1.00]

## Communities (60 total, 9 thin omitted)

### Community 0 - "Fusion Config & Core Types"
Cohesion: 0.06
Nodes (59): Enum, Detection, FusedDetection, ndarray, Regime, ndarray, Regime, ndarray (+51 more)

### Community 1 - "Frame Fusion Runtime"
Cohesion: 0.07
Nodes (34): FrameResult, main(), EO native box -> scale to IR size -> affine -> axis-aligned IR box., warp_box(), Detection, ndarray, PipelineConfig, Regime (+26 more)

### Community 2 - "Registration & Decision Concepts"
Cohesion: 0.07
Nodes (37): Affine transform (shift+zoom+stretch+rotate), IoU box-match metric, Cameras not co-boresighted (parallax/FOV), RANSAC robust fit, EO/IR Registration (affine transform), Resize does not affect IoU (ruled-out trap), Self-calibration (auto affine from detections), IoU association (EO box = IR box) (+29 more)

### Community 3 - "Affine Registration Fitting"
Cohesion: 0.11
Nodes (26): centers(), main(), Return list of (cx, cy) normalised centres in a YOLO label file., Detection, ndarray, apply_affine(), apply_homography(), AutoRegistrar (+18 more)

### Community 4 - "Kalman Tracking"
Cohesion: 0.11
Nodes (14): FusedDetection, ndarray, KalmanBoxTracker, Greedy IoU matching between predicted track boxes and detections., Advance one frame with the current fused detections.          Returns the confir, One-step predictions of confirmed tracks — priors for plan §5 feedback., state [cx,cy,w,h,...] -> [x1,y1,x2,y2]., Constant-velocity Kalman filter for one bounding box (plan §5).      State x = [ (+6 more)

### Community 5 - "Detection Metrics & Eval"
Cohesion: 0.11
Nodes (24): _ap_from_pr(), _best_f1(), evaluate(), _iou_matrix(), _match_image(), Self-contained detection metrics (numpy only).  Computes precision / recall / F1, Full evaluation over a dataset.      Returns a dict with overall + per-class pre, IoU between every box in a (M x4) and b (N x4) -> (M x N). (+16 more)

### Community 6 - "ProbEn Probabilistic Fusion"
Cohesion: 0.19
Nodes (18): Detection, FusedDetection, ndarray, main(), show(), ProbEn — Probabilistic Ensembling for multimodal detection fusion.  Chen et al.,, _cluster(), _distribution() (+10 more)

### Community 7 - "Dataset Reviewer (Bit-Flipper)"
Cohesion: 0.21
Nodes (16): commit(), extract_experiment_code(), find_twin(), flip_bit_generic(), get_history(), get_images(), images_list(), index() (+8 more)

### Community 8 - "Fuzzy Trust Engine"
Cohesion: 0.19
Nodes (11): _demo(), FuzzyTrust, FuzzyTrustConfig, Fuzzy sensor-trust engine — smooth EO/IR weighting (replaces the regime table)., Knobs for :class:`FuzzyTrust` — all tunable, none sacred.      The defaults are, Brightness + target size -> a smooth EO/IR trust weighting.      Stateless: the, Degree to which the inputs belong to each term (handy for debugging)., Fused trust in IR, in [0, 1] (0 = believe EO fully, 1 = believe IR).          Su (+3 more)

### Community 9 - "Dual-Sensor Frame Extraction"
Cohesion: 0.20
Nodes (14): extract_synced_frames(), get_common_id(), get_video_hash(), load_processed_hashes(), main(), Time based frame extractor for dual-sensor video datasets., Scans directories for matching Common IDs and processes them., Generate a hash based on file size and name to track processed files. (+6 more)

### Community 10 - "EO Low-Light Preprocessing"
Cohesion: 0.18
Nodes (11): ndarray, clahe(), enhance(), gamma_correct(), EO low-light preprocessing (plan §2.3 / §6 ``eo_preprocess``).  The active regim, Brighten an image with a gamma curve (gamma < 1 lifts shadows)., Contrast-limited adaptive histogram equalisation on the L (LAB) channel., Apply the named EO enhancement.      ``mode`` is one of ``None``/``"none"``, ``" (+3 more)

### Community 11 - "CVAT Stat Plotting"
Cohesion: 0.27
Nodes (13): bar_labels(), count_files(), extract_experiment_name(), extract_prefix_tokens(), main(), parse_file(), plot_categories(), plot_condition_sensor() (+5 more)

### Community 12 - "YOLO/SAHI Detector Wrapper"
Cohesion: 0.23
Nodes (7): Detection, ndarray, Detector, Pick a torch device string, honouring an explicit choice., One YOLO model behind a uniform ``run`` API, labelled EO or IR.      Parameters, Detect on a BGR frame at confidence ``conf``; return Detections.          ``conf, resolve_device()

### Community 13 - "SAHI Web Inference Viewer"
Cohesion: 0.28
Nodes (8): draw_visuals(), extract_key_and_frame(), get_sensor_type(), load_dataset(), load_sahi_model(), Loads the SAHI model only once to prevent lag when moving the slider., Scans directories and pairs Images with their Ground Truth Labels., Draws Ground Truth (Red) and SAHI Predictions (Green) on the image.

### Community 14 - "Mutual-Information Alignment"
Cohesion: 0.39
Nodes (7): align_mi(), joint_histogram(), main(), mi_cost(), mutual_information(), preprocess(), Convert to grayscale, apply CLAHE, and resize.

### Community 15 - "EO/IR Dataset Audit"
Cohesion: 0.39
Nodes (7): load_marked_files(), load_progress(), main(), Loads the progress for the specific folder from the JSON object., Updates the JSON file only if the new index is higher than the old one., save_marked_files(), save_progress_if_higher()

### Community 16 - "Synthetic Demo & Unit Checks"
Cohesion: 0.43
Nodes (6): _approx(), main(), Assert the core math matches the plan's formulas., Simulate a moving drone across day->twilight->night and track it., tracking_scenario(), unit_checks()

### Community 17 - "Frame Sync & Sizing"
Cohesion: 0.29
Nodes (6): ndarray, nearest_timestamp(), Frame sync + sizing (plan §2.1).  Temporal sync pairs EO and IR frames by timest, Put EO and IR into a shared pixel frame (resize only — no warping).      - If ``, Index of the candidate timestamp nearest ``target_ts`` within tolerance.      Re, sync_and_size()

### Community 18 - "YOLO Box Viewer"
Cohesion: 0.38
Nodes (6): draw_yolo_boxes(), extract_key_and_frame(), get_sensor_type(), load_dataset(), Scans directories and returns a dictionary of valid Image/Label pairs., Opens image, reads YOLO txt, and draws the bounding boxes.

### Community 20 - "Video Annotation & Relabel"
Cohesion: 0.47
Nodes (5): draw_custom_boxes(), process_video(), Modifies results in-place:      - Keeps 'bird' as 'bird'     - Changes ALL other, Draw bounding boxes and custom labels on frame., relabel_as_drone_or_bird()

### Community 21 - "Registration Visualization"
Cohesion: 0.60
Nodes (5): iou(), main(), read_label(), register_eo_box(), yolo_to_xyxy()

### Community 22 - "Dataset Stats"
Cohesion: 0.70
Nodes (4): count_files_by_category(), extract_prefix_tokens(), main(), plot_stats()

### Community 23 - "Manual Point Selection"
Cohesion: 0.40
Nodes (4): get_points(), Mouse callback function to record clicks., Helper to open window and collect 4 points., select_points()

### Community 24 - "Extraction Debug & Prep"
Cohesion: 0.50
Nodes (3): generate_insight_report(), normalize_name(), organize_dataset()

### Community 25 - "EO/IR Image Counting"
Cohesion: 0.83
Nodes (3): count_sensor_images(), main(), Path

### Community 26 - "Train/Test Split"
Cohesion: 0.83
Nodes (3): main(), Path, split_yolo_dataset()

## Knowledge Gaps
- **13 isolated node(s):** `ndarray`, `ndarray`, `opencv-python dependency`, `split-folders dependency`, `numpy dependency (fusion core)` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FusionPipeline` connect `Frame Fusion Runtime` to `Fusion Config & Core Types`, `Affine Registration Fitting`, `Kalman Tracking`, `Detection Metrics & Eval`, `Fuzzy Trust Engine`, `EO Low-Light Preprocessing`, `YOLO/SAHI Detector Wrapper`, `Synthetic Demo & Unit Checks`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Why does `main()` connect `Detection Metrics & Eval` to `Frame Fusion Runtime`, `ProbEn Probabilistic Fusion`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Why does `Detection` connect `Fusion Config & Core Types` to `Frame Fusion Runtime`, `Affine Registration Fitting`, `YOLO/SAHI Detector Wrapper`, `ProbEn Probabilistic Fusion`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 25 inferred relationships involving `Detection` (e.g. with `FrameResult` and `Detection`) actually correct?**
  _`Detection` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `FusedDetection` (e.g. with `FrameResult` and `Detection`) actually correct?**
  _`FusedDetection` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `FusionPipeline` (e.g. with `main()` and `tracking_scenario()`) actually correct?**
  _`FusionPipeline` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Regime` (e.g. with `Detection` and `FusedDetection`) actually correct?**
  _`Regime` has 15 INFERRED edges - model-reasoned connections that need verification._