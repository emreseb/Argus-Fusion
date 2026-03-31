from pathlib import Path

def count_sensor_images(img_dir):
    eo_count = 0
    ir_count = 0
    unknown_count = 0
    the_survivors_file = "list.txt"
    big_yahu_mode = False

    print(f"Scanning directory: {img_dir}...\n")

    if big_yahu_mode:
        f_out = open(the_survivors_file, 'w')
    else:
        f_out = None

    try:
        for img_path in Path(img_dir).rglob("*"):
            if img_path.suffix.lower() in ['.txt', '.jpg']: 
                
                # FIX: .stem is a property, not a callable method
                filename = img_path.stem
                
                # FIX: Removed redundant check
                if filename.startswith('.'):
                    continue
                
                if "_0_" in filename:
                    print(f"EO {filename}")
                    if f_out:
                        f_out.write(f"{filename}\n")
                    eo_count += 1
                    
                elif "_1_" in filename:
                    print(f"IR {filename}")
                    if f_out:
                        f_out.write(f"{filename}\n")
                    ir_count += 1
                
                else:
                    print(f"X  {filename}")
                    unknown_count += 1
                    
    finally:
        # Ensure the file is safely closed when done
        if f_out:
            f_out.close()

    print("--- File Count Complete ---")
    print(f"📷 EO Files (_0_): {eo_count}")
    print(f"🌡️ IR Files (_1_): {ir_count}")
    
    if unknown_count > 0:
        print(f"Unknown Files: {unknown_count}")
        
    print("-" * 28)
    print(f"Total Files Found: {eo_count + ir_count + unknown_count}")

def main():
    img_directory = "/home/emre/Desktop/NATO/DATASETv3/fully_paired_annotated/labels"
    
    if not Path(img_directory).exists():
        print(f"Error: Directory '{img_directory}' does not exist.")
        return
        
    count_sensor_images(img_directory)

if __name__ == "__main__":
    main()