import os
import re
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
SRC_DIR = r"D:\Teams 4th Year Spring semestar\Grad Project\DATASETv3 copy"

def normalize_name(name):
    # 1. Remove the formatting prefix if it exists
    core = re.sub(r'^\d_\d{3}_\d_', '', name)
    
    # 2. Normalize the word "frame"
    core = re.sub(r'_?frame_?', '_', core, flags=re.IGNORECASE)
    
    # 3. Strip leading zeros from the final frame number
    core = re.sub(r'_0*(\d+)$', r'_\1', core)
    
    return core.lower()

def generate_insight_report(src_dir):
    src = Path(src_dir)
    print(f"Scanning source directory: {src}")
    
    label_map = {}
    # 1. Gather all labels
    for txt_path in src.rglob('*.txt'):
        if txt_path.name in['classes.txt', 'train.txt']: 
            continue
            
        sensor = 'EO' if 'EO' in txt_path.parts else 'IR' if 'IR' in txt_path.parts else None
        if not sensor: 
            continue
            
        norm_key = normalize_name(txt_path.stem)
        label_map[(sensor, norm_key)] = txt_path.name

    matched_keys = set()
    unmatched_images =[]
    
    # 2. Gather images and check matches
    for img_path in list(src.rglob('*.jpg')) + list(src.rglob('*.png')):
        sensor = 'EO' if 'EO' in img_path.parts else 'IR' if 'IR' in img_path.parts else None
        if not sensor: 
            continue
        
        norm_key = normalize_name(img_path.stem)
        
        if (sensor, norm_key) in label_map:
            matched_keys.add((sensor, norm_key))
        else:
            # Keep track of what failed
            unmatched_images.append((sensor, img_path.name, norm_key))

    # Leftover labels that never found an image
    unmatched_labels = {k: v for k, v in label_map.items() if k not in matched_keys}

    # 3. Generate Report
    report_path = 'unmatched_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"--- UNMATCHED IMAGES SAMPLE (Total: {len(unmatched_images)}) ---\n")
        f.write(f"{'Sensor':<8} | {'Original Image Name':<35} | {'How the script reads it (Key)':<30}\n")
        f.write("-" * 80 + "\n")
        # Print ALL unmatched images now
        for item in unmatched_images: 
            f.write(f"{item[0]:<8} | {item[1]:<35} | {item[2]:<30}\n")
            
        f.write("\n\n")
        f.write(f"--- UNMATCHED LABELS SAMPLE (Total: {len(unmatched_labels)}) ---\n")
        f.write(f"{'Sensor':<8} | {'Original Label Name':<35} | {'How the script reads it (Key)':<30}\n")
        f.write("-" * 80 + "\n")
        # Print ALL leftover labels
        for k, orig_name in list(unmatched_labels.items()):
            f.write(f"{k[0]:<8} | {orig_name:<35} | {k[1]:<30}\n")

    print(f"\n✅ Done! Created '{report_path}' in your current folder.")
    print("Please paste a few lines from both the Images and Labels lists here so we can see the final mismatch!")

if __name__ == "__main__":
    generate_insight_report(SRC_DIR)