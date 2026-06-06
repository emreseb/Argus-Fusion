# DATASET-TOOLS

A collection of Python tools for building, inspecting, and evaluating object-detection datasets (YOLO format, with EO/IR imagery).

## What it's for

This repository gathers the scripts used to take raw images and labels and turn them into a clean, trainable dataset — and to train and check models on it. The main jobs it handles:

- **Viewing** images alongside their YOLO labels to spot-check the data.
- **Reviewing / fixing labels** — flipping, correcting, and cleaning annotations.
- **Dataset stats** — counting classes, conditions, and experiments, and plotting them.
- **Splitting** data into train/val sets.
- **Training** YOLO models (CPU/GPU trainers and configs in `yaml-files/`).
- **Inference & evaluation** — running models on images, including tiled (SAHI) inference.

## Layout

| Path | Purpose |
|------|---------|
| `view_all.py` | Streamlit viewer for images + labels |
| `bit-flipper/` | Flask tool for reviewing and fixing labels |
| `trainers/` | YOLO training scripts |
| `live/` | Inference tools (incl. SAHI tiled inference) |
| `model-eval/` | Model evaluation scripts |
| `yaml-files/` | Dataset/training configs |
| `CVAT-Stats/` | Dataset statistics and plots |

## Running the tools

Most tools read your dataset from environment variables instead of hardcoded paths:

- `DATASET_ROOT` — dataset root (expects `images/` and `labels/` underneath)
- `MODEL_PATH` — path to a YOLO model (for inference/eval)
- `SAHI_DEVICE` — `cpu` (default), `mps`, or `cuda`

Example:

```bash
export DATASET_ROOT="/path/to/dataset"
python -m streamlit run view_all.py
```

The dataset images and labels are **not** stored in git — point the tools at a local copy using the variables above.
