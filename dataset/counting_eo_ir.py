import shutil
from pathlib import Path

def count_sensor_images(img_dir: Path, eo_dest: Path, ir_dest: Path):
    eo_count = 0
    ir_count = 0
    unknown_count = 0

    print(f"Scanning directory: {img_dir}...\n")

    # Create destination directories
    eo_dest.mkdir(parents=True, exist_ok=True)
    ir_dest.mkdir(parents=True, exist_ok=True)

    for img_path in img_dir.rglob("*"):
        if eo_dest in img_path.parents or ir_dest in img_path.parents:
            continue

        # Check if it's actually a file, and matches your extensions
        if img_path.is_file() and img_path.suffix.lower() in ['.txt', '.jpg']: 
            
            # Check the full name for hidden files
            if img_path.name.startswith('.'):
                continue
                
            filename = img_path.stem
            
            if "_0_" in filename:
                print(f"EO {filename}")
                shutil.copy(img_path, eo_dest)
                eo_count += 1
                
            elif "_1_" in filename:
                print(f"IR {filename}")
                shutil.copy(img_path, ir_dest)
                ir_count += 1
                
            else:
                print(f"X  {filename}")
                unknown_count += 1
                
    print("\n--- File Count Complete ---")
    print(f"📷 EO Files (_0_): {eo_count}")
    print(f"🌡️ IR Files (_1_): {ir_count}")
    
    if unknown_count > 0:
        print(f"❓ Unknown Files: {unknown_count}")
        
    print("-" * 28)
    print(f"Total Files Processed: {eo_count + ir_count + unknown_count}")


def main():
    base_dir = Path("/home/emre/Desktop/NATO/DATASETv3/labels(removed empty txts)/obj_train_data")
    
    eo_dest = Path("/home/emre/Desktop/NATO/DATASETv3/training-eo-ir-seperated/EO-labels")
    ir_dest = Path("/home/emre/Desktop/NATO/DATASETv3/training-eo-ir-seperated/IR-labels")
    
    if not base_dir.exists():
        print(f"Error: Directory '{base_dir}' does not exist.")
        return
        
    count_sensor_images(base_dir, eo_dest, ir_dest)


if __name__ == "__main__":
    main()