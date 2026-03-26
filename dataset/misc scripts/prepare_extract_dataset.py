import os
import shutil
import re
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
SRC_DIR = r"D:\Teams 4th Year Spring semestar\Grad Project\DATASETv3 copy"
DEST_DIR = r"D:\Teams 4th Year Spring semestar\Grad Project\Set_V3_Extracted_Matching_Names"

# CHANGE THIS TO 'True' LATER WHEN YOU ARE READY TO TRAIN YOLO!
# YOLO requires the label file to have the exact same base name as the image.
RENAME_LABEL_TO_MATCH_IMAGE = True  
# ==========================================

def organize_dataset(src_dir, dest_dir):
    src = Path(src_dir)
    dest = Path(dest_dir)
    
    # Create the flattened structure for EO and IR
    for sensor in ['EO', 'IR']:
        (dest / sensor / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (dest / sensor / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        
    def normalize_name(name):
        """
        Creates a 'core' string to pair images and labels safely.
        Fixes the zero-padding discrepancy: e.g., 'frame000001' vs '_1'
        """
        # 1. Remove the formatting prefix if it exists (e.g., 1_001_0_)
        core = re.sub(r'^\d_\d{3}_\d_', '', name)
        
        # 2. Normalize the word "frame" and its surrounding underscores into a single underscore
        core = re.sub(r'_?frame_?', '_', core, flags=re.IGNORECASE)
        
        # 3. FIX: Strip leading zeros from the final frame number!
        # This converts things like '_000012' into '_12' and '_000000' into '_0'
        # It finds an underscore, any number of zeros, and captures the remaining digits at the end.
        core = re.sub(r'_0*(\d+)$', r'_\1', core)
        
        return core.lower()

    print(f"Scanning source directory: {src}")
    print("Gathering labels...")
    label_map = {}
    
    # Recursively find all .txt files
    for txt_path in src.rglob('*.txt'):
        if txt_path.name in ['classes.txt', 'train.txt']: 
            continue
            
        sensor = 'EO' if 'EO' in txt_path.parts else 'IR' if 'IR' in txt_path.parts else None
        if not sensor: 
            continue
            
        # Create a normalized key mapped to the original file path
        norm_key = normalize_name(txt_path.stem)
        label_map[(sensor, norm_key)] = txt_path

    print(f"Found {len(label_map)} valid label files. Matching images...")
    
    matched_count = 0
    missing_count = 0
    
    # Recursively find all images
    for img_path in list(src.rglob('*.jpg')) + list(src.rglob('*.png')):
        sensor = 'EO' if 'EO' in img_path.parts else 'IR' if 'IR' in img_path.parts else None
        if not sensor: 
            continue
        
        norm_key = normalize_name(img_path.stem)
        
        # Look up the normalized key in our label map
        if (sensor, norm_key) in label_map:
            txt_path = label_map[(sensor, norm_key)]
            
            # --- COPY IMAGE --- (Always keeps original name)
            new_img_path = dest / sensor / 'images' / 'train' / img_path.name
            shutil.copy2(img_path, new_img_path)
            
            # --- COPY LABEL ---
            if RENAME_LABEL_TO_MATCH_IMAGE:
                # Renames label to match the image name precisely (Required for YOLO)
                new_lbl_name = f"{img_path.stem}.txt"
            else:
                # Keeps the label's original name exactly as it is (As requested)
                new_lbl_name = txt_path.name
                
            new_txt_path = dest / sensor / 'labels' / 'train' / new_lbl_name
            shutil.copy2(txt_path, new_txt_path)
            
            matched_count += 1
        else:
            missing_count += 1

    print(f"\n✅ Success! Copied and matched {matched_count} image-label pairs to {dest}")
    if missing_count > 0:
        print(f"⚠️ Skipped {missing_count} images because no matching .txt label was found.")

if __name__ == "__main__":
    organize_dataset(SRC_DIR, DEST_DIR)