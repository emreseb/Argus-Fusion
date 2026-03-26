import os
import re

# --- Configuration ---
images_dir = "/Users/emre/Desktop/DATASETv3/images" 
labels_dir = "/Users/emre/Desktop/DATASETv3/labels/obj_train_data" 

def extract_key_and_frame(filename):
    """
    Extracts the key (e.g., B20, E9) and the frame number (e.g., 000011 or 11).
    """
    match = re.search(r'([A-Z]+\d+).*?(\d+)$', filename)
    if match:
        key = match.group(1)
        frame = int(match.group(2))
        return key, frame
    return None, None

def get_sensor_type(root_path, filename):
    """
    Determines if the file belongs to the EO or IR sensor stream.
    Checks the folder path first, then falls back to checking the filename prefix.
    """
    # Standardize slashes to catch the folder names easily
    path_str = root_path.replace('\\', '/')
    
    # 1. Check if it is sitting in an EO or IR folder
    if '/EO/' in path_str or path_str.endswith('/EO'): 
        return 'EO'
    if '/IR/' in path_str or path_str.endswith('/IR'): 
        return 'IR'
    
    # 2. Fallback: check the filename itself (_0_ is EO, _1_ is IR)
    if '_0_' in filename: 
        return 'EO'
    if '_1_' in filename: 
        return 'IR'
        
    return 'UNKNOWN'

def sync_filenames():
    # 1. Map the 'Full Names' from the images
    # Key: (Sensor, Identifier, Frame) -> Value: Full Name without extension
    full_name_map = {}
    
    print(f"Recursively mapping image names in: {images_dir}")
    for root, _, files in os.walk(images_dir):
        for img_file in files:
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                name_no_ext = os.path.splitext(img_file)[0]
                key, frame = extract_key_and_frame(name_no_ext)
                sensor = get_sensor_type(root, name_no_ext)
                
                if key and frame is not None:
                    # Now storing it with the 3-part lock
                    full_name_map[(sensor, key, frame)] = name_no_ext

    print(f"\nRecursively renaming label files in: {labels_dir}")
    rename_count = 0
    missing_count = 0
    regex_fail_count = 0

    for root, _, files in os.walk(labels_dir):
        for label_file in files:
            if label_file.endswith('.txt'):
                label_no_ext = os.path.splitext(label_file)[0]
                key, frame = extract_key_and_frame(label_no_ext)
                sensor = get_sensor_type(root, label_no_ext)
                
                if key and frame is not None:
                    # Match using the Sensor + Key + Frame
                    if (sensor, key, frame) in full_name_map:
                        new_name = full_name_map[(sensor, key, frame)] + ".txt"
                        
                        old_path = os.path.join(root, label_file)
                        new_path = os.path.join(root, new_name)
                        
                        if old_path != new_path:
                            os.rename(old_path, new_path)
                            rename_count += 1
                    else:
                        print(f"⚠️ NO IMAGE MATCH: '{label_file}' (Sensor: {sensor}, Key: {key}, Frame: {frame})")
                        missing_count += 1
                else:
                    print(f"🛑 REGEX FAILED: '{label_file}' does not match pattern.")
                    regex_fail_count += 1

    print(f"\n--- Process Complete ---")
    print(f"✅ Renamed: {rename_count} labels")
    print(f"⚠️ Skipped (No matching image): {missing_count}")
    print(f"🛑 Skipped (Failed regex): {regex_fail_count}")

if __name__ == "__main__":
    sync_filenames()