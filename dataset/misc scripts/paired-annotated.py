import os
import shutil
from pathlib import Path

"""
Combined EO/IR Twin + Annotation Validator
-------------------------------------------
A frame set is only accepted if ALL of the following are true:
  1. The EO image (_0_) exists
  2. The IR twin (_1_) exists
  3. The EO image has a matching .txt annotation
  4. The IR image has a matching .txt annotation

Rejected if ANY of the above is missing.
"""

# --- Configuration ---
images_dir      = "/home/emre/Desktop/NATO/DATASETv3/images(all)"
labels_dir      = "/home/emre/Desktop/NATO/DATASETv3/labels(removed empty txts)/obj_train_data"
output_base_dir = "/home/emre/Desktop/NATO/DATASETv3/fully_paired_annotated"

DEBUG_MODE = True
COPY_FILES = True   # Set to False for a dry-run report only


def main():
    if DEBUG_MODE:
        print(f"Scanning images in : {images_dir}")
        print(f"Scanning labels in : {labels_dir}\n")

    # ------------------------------------------------------------------ #
    # 1. Collect EO / IR images keyed by their "universal" stem           #
    #    Universal key: replace _0_ or _1_ with _X_ so twins share a key  #
    # ------------------------------------------------------------------ #
    eo_images = {}   # universal_key -> Path
    ir_images = {}

    for img_path in Path(images_dir).rglob("*.jpg"):
        if img_path.name.startswith("."):
            continue
        name = img_path.name
        if "_0_" in name:
            eo_images[name.replace("_0_", "_X_")] = img_path
        elif "_1_" in name:
            ir_images[name.replace("_1_", "_X_")] = img_path

    # ------------------------------------------------------------------ #
    # 2. Collect all annotation stems (.txt)                              #
    # ------------------------------------------------------------------ #
    txt_stems = {p.stem for p in Path(labels_dir).rglob("*.txt")
                 if not p.name.startswith(".")}
    txt_paths = {p.stem: p for p in Path(labels_dir).rglob("*.txt")
                 if not p.name.startswith(".")}

    # ------------------------------------------------------------------ #
    # 3. Evaluate every EO/IR pair against annotation requirements        #
    # ------------------------------------------------------------------ #
    eo_keys = set(eo_images.keys())
    ir_keys = set(ir_images.keys())
    all_keys = eo_keys | ir_keys

    fully_valid   = []   # (universal_key) — all 4 checks pass
    rejected      = []   # (universal_key, reason)

    for key in sorted(all_keys):
        eo_path = eo_images.get(key)
        ir_path = ir_images.get(key)

        missing = []

        if eo_path is None:
            missing.append("EO image missing")
        if ir_path is None:
            missing.append("IR twin missing")

        if eo_path and eo_path.stem not in txt_stems:
            missing.append(f"annotation missing for EO ({eo_path.name})")
        if ir_path and ir_path.stem not in txt_stems:
            missing.append(f"annotation missing for IR ({ir_path.name})")

        if missing:
            rejected.append((key, missing))
        else:
            fully_valid.append(key)

    # ------------------------------------------------------------------ #
    # 4. Debug report                                                     #
    # ------------------------------------------------------------------ #
    if DEBUG_MODE:
        print(f"Total EO images found  : {len(eo_images)}")
        print(f"Total IR images found  : {len(ir_images)}")
        print(f"Total annotation files : {len(txt_stems)}")
        print("-" * 65)
        print(f"Fully valid pairs      : {len(fully_valid)}  ✓")
        print(f"Rejected frame sets    : {len(rejected)}  ✗")

        if rejected:
            print("\nRejected details (first 20):")
            for key, reasons in rejected[:20]:
                print(f"  [{key}]")
                for r in reasons:
                    print(f"      • {r}")
            if len(rejected) > 20:
                print(f"  ... and {len(rejected) - 20} more.")

        print()

    # ------------------------------------------------------------------ #
    # 5. Copy valid sets to output directory                              #
    # ------------------------------------------------------------------ #
    if COPY_FILES and fully_valid:
        out_img_dir = os.path.join(output_base_dir, "images")
        out_txt_dir = os.path.join(output_base_dir, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_txt_dir, exist_ok=True)

        print("-" * 65)
        print(f"Copying {len(fully_valid)} valid pairs "
              f"({len(fully_valid) * 2} images + {len(fully_valid) * 2} labels) to:")
        print(f"  Images -> {out_img_dir}")
        print(f"  Labels -> {out_txt_dir}\n")

        for key in fully_valid:
            eo_path = eo_images[key]
            ir_path = ir_images[key]

            shutil.copy(eo_path, out_img_dir)
            shutil.copy(ir_path, out_img_dir)
            shutil.copy(txt_paths[eo_path.stem], out_txt_dir)
            shutil.copy(txt_paths[ir_path.stem], out_txt_dir)

        print("Copy complete.")

    elif not COPY_FILES:
        print("-" * 65)
        print("COPY_FILES is False — dry run only, no files moved.")


if __name__ == "__main__":
    errors = []
    if not Path(images_dir).exists():
        errors.append(f"Images dir not found : {images_dir}")
    if not Path(labels_dir).exists():
        errors.append(f"Labels dir not found : {labels_dir}")
    if errors:
        for e in errors:
            print(f"Error: {e}")
    else:
        main()