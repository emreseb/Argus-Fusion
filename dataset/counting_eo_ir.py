from pathlib import Path

"""
This script counts how many EO and IR images are in a given directory based on 
their filename patterns

"""


def count_sensor_images(img_dir):
    eo_count = 0
    ir_count = 0
    unknown_count = 0

    print(f"Scanning directory: {img_dir}...\n")

    for img_path in Path(img_dir).rglob("*"):
        
        # Only look at image files
        if img_path.suffix.lower() in ['.txt','.jpg']: #.is_dir() if you want directory count
            filename = img_path.name
            
            # --- THE FIX ---
            # Skip any macOS hidden metadata files
            if filename.startswith('.') or filename.startswith('.'):
                continue
            
            # Tally based on the sensor identifier
            if "_0_" in filename:
                eo_count += 1
            elif "_1_" in filename:
                ir_count += 1
            else:
                unknown_count += 1

    print("--- Image Count Complete ---")
    print(f"📷 EO Images (_0_): {eo_count}")
    print(f"🌡️ IR Images (_1_): {ir_count}")
    
    if unknown_count > 0:
        print(f"⚠️ Unknown Images: {unknown_count}")
        
    print("-" * 28)
    print(f"Total Images Found: {eo_count + ir_count + unknown_count}")

def main():
    img_directory = "/home/emre/Desktop/NATO/DATASETv3/twin_eo_ir"
    
    if not Path(img_directory).exists():
        print(f"❌ Error: Image directory '{img_directory}' does not exist.")
        return
        
    count_sensor_images(img_directory)

if __name__ == "__main__":
    main()