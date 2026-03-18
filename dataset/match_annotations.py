import os
import re

# --- Configuration ---
# Update these to the folders where your files are currently located
images_dir = "Users/emre/Desktop/DATASETv3/images" # Contains files like 1_100_1_B20_000000.jpg
labels_dir = "Users/emre/Desktop/DATASETv3/labels" # Contains files like B20_Frame_000000.txt

def extract_key_and_frame(filename):
    """
    Extracts the key (e.g., B20) and the frame number (e.g., 000000).
    """
    # This regex looks for the identifier (Letter + Numbers) followed by a frame number
    # It works for both '1_100_1_B20_000000' and 'B20_Frame_000000'
    match = re.search(r'([A-Z]\d+).*?(\d{6})', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def sync_filenames():
    # 1. Map the 'Full Names' from the images
    # Key: (Identifier, Frame) -> Value: Full Name without extension
    full_name_map = {}
    
    print(f"Recursively mapping image names in: {images_dir}")
    # os.walk scans the root directory and all subdirectories
    for root, _, files in os.walk(images_dir):
        for img_file in files:
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                name_no_ext = os.path.splitext(img_file)[0]
                key, frame = extract_key_and_frame(name_no_ext)
                if key and frame:
                    full_name_map[(key, frame)] = name_no_ext

    print(f"Recursively renaming label files in: {labels_dir}")
    rename_count = 0
    missing_count = 0

    # os.walk again for the labels
    for root, _, files in os.walk(labels_dir):
        for label_file in files:
            if label_file.endswith('.txt'):
                label_no_ext = os.path.splitext(label_file)[0]
                key, frame = extract_key_and_frame(label_no_ext)
                
                if (key, frame) in full_name_map:
                    new_name = full_name_map[(key, frame)] + ".txt"
                    
                    # Use 'root' instead of labels_dir to rename it in its exact subfolder
                    old_path = os.path.join(root, label_file)
                    new_path = os.path.join(root, new_name)
                    
                    # Prevent error if the file is already named correctly
                    if old_path != new_path:
                        os.rename(old_path, new_path)
                        rename_count += 1
                else:
                    missing_count += 1

    print(f"--- Process Complete ---")
    print(f"Successfully renamed: {rename_count} labels")
    print(f"Skipped (no match found): {missing_count}")

if __name__ == "__main__":
    sync_filenames()