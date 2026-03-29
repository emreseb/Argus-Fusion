import os
import shutil
from pathlib import Path

"""
This script matches .jpg images with their corresponding .txt labels.
It features a DEBUG_MODE for reporting and a MOVE_FILES boolean to 
execute the transfer into a clean, paired directory structure.
"""

# --- Configuration ---
images_dir = "/home/emre/Desktop/NATO/DATASETv3/twin_eo_ir"
labels_dir = "/home/emre/Desktop/NATO/DATASETv3/labels(removed empty txts)/obj_train_data"
output_base_dir = "/home/emre/Desktop/NATO/DATASETv3/pairtxtimg"

DEBUG_MODE = True
MOVE_FILES = False  # Set to True to transfer the matched files

def check_and_transfer_pairs():
    if DEBUG_MODE:
        print(f"Scanning images in: {images_dir}")
        print(f"Scanning labels in: {labels_dir}\n")

    # 1. Gather all files. 
    # .stem extracts just the filename without the .jpg or .txt extension.
    img_paths = {img.stem: img for img in Path(images_dir).rglob("*.jpg") if not img.name.startswith('.')}
    txt_paths = {txt.stem: txt for txt in Path(labels_dir).rglob("*.txt") if not txt.name.startswith('.')}

    # 2. Use Set Math on the stems to instantly find pairs and mismatches
    img_stems = set(img_paths.keys())
    txt_stems = set(txt_paths.keys())

    paired_stems = img_stems.intersection(txt_stems)
    missing_txt = img_stems - txt_stems
    missing_img = txt_stems - img_stems

    # 3. Debug Output
    if DEBUG_MODE:
        print(f"Total Images Found: {len(img_stems)}")
        print(f"Total Labels Found: {len(txt_stems)}")
        print("-" * 60)
        
        print(f"Successfully Paired: {len(paired_stems)} complete sets\n")

        if missing_txt:
            print(f"Found {len(missing_txt)} images missing their .txt label:")
            for stem in list(missing_txt)[:10]:
                print(f"   - {stem}.jpg has no label.")
            if len(missing_txt) > 10:
                print(f"   ... and {len(missing_txt) - 10} more.")
            print()

        if missing_img:
            print(f"Found {len(missing_img)} labels missing their .jpg image:")
            for stem in list(missing_img)[:10]:
                print(f"   - {stem}.txt has no image.")
            if len(missing_img) > 10:
                print(f"   ... and {len(missing_img) - 10} more.")
            print()

    # 4. Transfer files if the boolean is True
    if MOVE_FILES and paired_stems:
        out_img_dir = os.path.join(output_base_dir, "images")
        out_txt_dir = os.path.join(output_base_dir, "paired_txt")

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_txt_dir, exist_ok=True)

        print("-" * 60)
        print(f"Transferring {len(paired_stems)} valid pairs to:")
        print(f"Images -> {out_img_dir}")
        print(f"Labels -> {out_txt_dir}")

        for stem in paired_stems:
            shutil.copy(img_paths[stem], out_img_dir)
            shutil.copy(txt_paths[stem], out_txt_dir)

        print("\nTransfer complete.")
        
    elif not MOVE_FILES:
        print("-" * 60)
        print("MOVE_FILES is set to False. No files were transferred.")

if __name__ == "__main__":
    if not Path(images_dir).exists():
        print(f"Error: Images directory '{images_dir}' does not exist.")
    elif not Path(labels_dir).exists():
        print(f"Error: Labels directory '{labels_dir}' does not exist.")
    else:
        check_and_transfer_pairs()