import os
import shutil
from pathlib import Path

"""
This script checks if every EO (_0_) image has its exact IR (_1_) twin,
and vice versa, down to the exact frame number.
It automatically ignores macOS hidden metadata files (._)
and copies all successful pairs to a new directory.
"""

images_dir = "/home/emre/Desktop/NATO/DATASETv3/images(all)"
output_dir = "/home/emre/Desktop/NATO/DATASETv3/twin_eo_ir"

def check_and_copy_pairs():
    print(f"Scanning for EO/IR image pairs in: {images_dir}\n")
    
    eo_images = {}
    ir_images = {}
    
    # 1. Gather all images and create their Universal Keys
    for img_path in Path(images_dir).rglob("*.jpg"):
        filename = img_path.name
        
        # Skip hidden metadata files
        if filename.startswith('.'):
            continue
            
        # Sort into EO and IR dictionaries, storing the full Path object
        if "_0_" in filename:
            universal_key = filename.replace("_0_", "_X_")
            eo_images[universal_key] = img_path
            
        elif "_1_" in filename:
            universal_key = filename.replace("_1_", "_X_")
            ir_images[universal_key] = img_path

    # 2. Compare the sets of Universal Keys
    eo_keys = set(eo_images.keys())
    ir_keys = set(ir_images.keys())
    
    paired_keys = eo_keys.intersection(ir_keys)
    missing_ir = eo_keys - ir_keys  # EO exists, missing IR twin
    missing_eo = ir_keys - eo_keys  # IR exists, missing EO twin
    
    # 3. Print the report
    print(f"Total EO Images Found: {len(eo_images)}")
    print(f"Total IR Images Found: {len(ir_images)}")
    print("-" * 60)
    
    if not missing_ir and not missing_eo:
        print(f"PERFECT MATCH! All {len(paired_keys)} pairs are complete and symmetrical.")
    else:
        print(f"Successfully Paired: {len(paired_keys)} sets\n")
        
        if missing_ir:
            print(f"Found {len(missing_ir)} EO images missing their IR twin:")
            for key in list(missing_ir)[:10]:
                # .name extracts just the filename from the stored Path object
                eo_name = eo_images[key].name
                expected_twin = eo_name.replace("_0_", "_1_")
                print(f"   - {eo_name}  -->  Missing: {expected_twin}")
            if len(missing_ir) > 10:
                print(f"   ... and {len(missing_ir) - 10} more.")
            print()
            
        if missing_eo:
            print(f"Found {len(missing_eo)} IR images missing their EO twin:")
            for key in list(missing_eo)[:10]:
                ir_name = ir_images[key].name
                expected_twin = ir_name.replace("_1_", "_0_")
                print(f"   - {ir_name}  -->  Missing: {expected_twin}")
            if len(missing_eo) > 10:
                print(f"   ... and {len(missing_eo) - 10} more.")

    # 4. Copy the successful pairs to the new directory
    if paired_keys:
        print("-" * 60)
        print(f"Copying {len(paired_keys)} valid pairs ({len(paired_keys) * 2} total images) to:")
        print(f"{output_dir}")
        
        # exist_ok=True prevents the script from crashing if the folder already exists
        os.makedirs(output_dir, exist_ok=True)
        
        for key in paired_keys:
            shutil.copy(eo_images[key], output_dir)
            shutil.copy(ir_images[key], output_dir)
            
        print("\nCopy operation complete.")

if __name__ == "__main__":
    if not Path(images_dir).exists():
        print(f"Error: Images directory '{images_dir}' does not exist.")
    else:
        check_and_copy_pairs()