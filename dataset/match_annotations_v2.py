import os
import re

# --- Configuration ---
images_dir = "/Users/emre/Desktop/DATASETv3/images/" 
labels_dir = "/Users/emre/Desktop/DATASETv3/labels/obj_train_data/ROMA_RENAMED" 

def extract_key_and_frame(filename):
    """
    Extracts the key (e.g., B20, ERF19) and the frame number (e.g., 000000 or 0).
    """
    # [A-Z]+\d+  -> Matches 1 or more letters followed by digits (e.g., B20, ERF19)
    # .*?        -> Matches any characters in between
    # (\d+)$     -> Matches 1 or more digits right at the end of the filename
    match = re.search(r'([A-Z]+\d+).*?(\d+)$', filename)
    if match:
        key = match.group(1)
        # Convert the frame string to an integer so '000000' and '0' both become exactly 0
        frame = int(match.group(2))
        return key, frame
    return None, None

def sync_filenames():
    full_name_map = {}
    
    print(f"Recursively mapping image names in: {images_dir}")
    for root, _, files in os.walk(images_dir):
        for img_file in files:
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                name_no_ext = os.path.splitext(img_file)[0]
                key, frame = extract_key_and_frame(name_no_ext)
                if key and frame is not None: # Check 'is not None' because frame could be 0
                    full_name_map[(key, frame)] = name_no_ext

    print(f"\nRecursively renaming label files in: {labels_dir}")
    rename_count = 0
    missing_count = 0
    regex_fail_count = 0

    for root, _, files in os.walk(labels_dir):
        for label_file in files:
            if label_file.endswith('.txt'):
                label_no_ext = os.path.splitext(label_file)[0]
                key, frame = extract_key_and_frame(label_no_ext)
                
                # Check 1: Did the regex successfully find a Key and Frame?
                if key and frame is not None:
                    # Check 2: Does this Key/Frame exist in our images map?
                    if (key, frame) in full_name_map:
                        new_name = full_name_map[(key, frame)] + ".txt"
                        
                        old_path = os.path.join(root, label_file)
                        new_path = os.path.join(root, new_name)
                        
                        if old_path != new_path:
                            os.rename(old_path, new_path)
                            rename_count += 1
                    else:
                        print(f"NO IMAGE MATCH: '{label_file}' (Looking for Image with Key: {key}, Frame: {frame})")
                        missing_count += 1
                else:
                    print(f"REGEX FAILED: '{label_file}' does not match the expected naming pattern.")
                    regex_fail_count += 1

    print(f"\n--- Process Complete ---")
    print(f" Renamed: {rename_count} labels")
    print(f" Skipped (No matching image): {missing_count}")
    print(f" Skipped (Failed regex pattern): {regex_fail_count}")

if __name__ == "__main__":
    sync_filenames()