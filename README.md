# DATASET-TOOLS

This repository is primarily a set of Python utilities for dataset auditing, visualization, labeling/review, and model-inference helpers.

## What’s “online” on Vercel

This repo includes a small static homepage (`index.html`) and `vercel.json` so you can connect the repo to Vercel and get:

- **A stable project homepage** (quick links + run instructions)
- **Preview deployments for every PR**, which makes collaboration/review much easier

Important: the interactive Python apps in this repo (Streamlit + Flask) are meant to run **locally** (or on a Python-native host). They depend on access to your dataset files on disk and, in some cases, ML dependencies that aren’t a good fit for Vercel’s serverless runtime.

## Local development (Python tools)

### Required environment variables

Most interactive tools now use environment variables instead of hardcoded machine paths.

- Copy `.env.example` to `.env` (optional) and export variables in your shell.

- **`DATASET_ROOT`**: root directory of your dataset (expects `images/` and `labels/` under it), or set the two vars below.
- **`DATASET_IMAGES_DIR`**: folder containing images.
- **`DATASET_LABELS_DIR`**: folder containing YOLO label `.txt` files (often `labels/obj_train_data`).

Optional:

- **`MODEL_PATH`**: path to a YOLO model (used by SAHI evaluator).
- **`SAHI_DEVICE`**: `"cpu"` (default), `"mps"`, or `"cuda"` depending on your machine.

### Streamlit: dataset viewer

```bash
export DATASET_ROOT="/path/to/DATASETv3"
python -m streamlit run view_all.py
```

### Streamlit: SAHI evaluator

```bash
export DATASET_ROOT="/path/to/DATASETv3"
export MODEL_PATH="/path/to/best.pt"
export SAHI_DEVICE="cpu"
python -m streamlit run live/picture-inf-web-sahi.py
```

### Flask: dataset reviewer (bit flipper)

```bash
export DATASET_ROOT="/path/to/DATASETv3"
python bit-flipper/bit-flipper.py
```

## Deploy to Vercel (team collaboration)

### One-time setup

1. Import the Git repo into Vercel.
2. Vercel will auto-detect this as a static site (served from `index.html`).

### Recommended workflow for a team

- **PRs for all changes**: Vercel automatically creates a **Preview Deployment** per PR.
- **Link PR → preview URL**: reviewers can open the preview without running anything.
- **Use env vars in scripts**: avoid hardcoded paths so teammates can run tools locally.

## Notes / limitations

- The repo does **not** currently include the dataset images/labels in git (by design). The interactive apps will show “no images found” until you point them at a local dataset path using the env vars above.
- If you want the interactive apps “online”, the usual approach is to host them on a Python-native platform (e.g., Streamlit Community Cloud / Render / a small VM) and keep Vercel for the landing page + previews.

