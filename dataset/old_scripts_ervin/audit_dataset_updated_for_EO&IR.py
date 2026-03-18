import cv2
import os
import glob
import sys
import json
import pyperclip 

# ================= CONFIGURATION =================
# Root path containing the newly extracted YOLO dataset
DATASET_ROOT = r"D:\Teams 4th Year Spring semestar\Grad Project\Set_V3_Extracted_Matching_Names"

# File where marked filenames will be saved
DELETE_LIST_FILE = "files_to_delete_V3_Extracted.txt"

# Single file to remember your position across EO and IR folders
PROGRESS_FILE = "audit_progress_V3_Extracted.json"

# Class configurations
COLORS = {
    0: (0, 255, 0),    # DRONE (Green)
    1: (0, 0, 255),    # BIRD (Red)
    2: (255, 0, 0),    # AIRPLANE (Blue)
    3: (0, 255, 255)   # HELICOPTER (Yellow)
}
CLASS_NAMES = {0: "DRONE", 1: "BIRD", 2: "AIRPLANE", 3: "HELICOPTER"}
# =================================================

def load_marked_files():
    if not os.path.exists(DELETE_LIST_FILE):
        return set()
    with open(DELETE_LIST_FILE, 'r') as f:
        return set(line.strip() for line in f.readlines())

def save_marked_files(marked_set):
    with open(DELETE_LIST_FILE, 'w') as f:
        for filename in marked_set:
            f.write(f"{filename}\n")

def load_progress(subset_choice):
    """Loads the progress for the specific folder from the JSON object."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
                return data.get(subset_choice, 0)
        except (ValueError, json.JSONDecodeError):
            return 0
    return 0

def save_progress_if_higher(new_index, subset_choice):
    """Updates the JSON file only if the new index is higher than the old one."""
    data = {}
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
        except (ValueError, json.JSONDecodeError):
            data = {}
            
    current_saved = data.get(subset_choice, 0)
    
    if new_index > current_saved:
        data[subset_choice] = new_index
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(data, f, indent=4) 

def main():
    print(f"Dataset Root: {DATASET_ROOT}")
    print("Available sensor folders to audit: EO, IR")
    
    # 1. Ask the user which sensor to audit
    subset_choice = input("Enter the sensor you want to audit (EO or IR): ").strip().upper()
    
    valid_choices = ['EO', 'IR']
    if subset_choice not in valid_choices:
        print(f"Invalid choice. Please run the script again and choose from {valid_choices}.")
        return

    # 2. Gather all images ONLY for the selected sensor
    all_images =[]
    print(f"\nScanning '{subset_choice}' directory...")
    
    # New YOLO Structure paths
    img_dir = os.path.join(DATASET_ROOT, subset_choice, "images", "train")
    label_dir = os.path.join(DATASET_ROOT, subset_choice, "labels", "train")
        
    if os.path.exists(img_dir):
        files = glob.glob(os.path.join(img_dir, "*.*"))
        for f in files:
            # Filter for image extensions
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                all_images.append((f, subset_choice, os.path.basename(f)))
    else:
        print(f"Error: Image directory not found at {img_dir}")
        return
        
    if not all_images:
        print(f"No images found in {img_dir}. Check your folder structure.")
        return

    # Sort images by filename to ensure deterministic order
    all_images.sort(key=lambda x: x[2])

    total_images = len(all_images)
    
    # 3. Load existing progress for THIS specific folder
    start_index = load_progress(subset_choice)
    
    if start_index >= total_images:
        start_index = total_images - 1
        
    print(f"Total Images in '{subset_choice}': {total_images}")
    print(f"Resuming at index: {start_index + 1} (Furthest Point Reached)")
    
    # 4. Load marked files
    marked_files = load_marked_files()
    print(f"Previously marked for deletion (across all folders): {len(marked_files)}")

    print("\n=== CONTROLS ===")
    print(" [D] or [SPACE] : Next Image")
    print("[A]            : Previous Image")
    print(" [C]            : Copy Filename")
    print("[X]            : MARK/UNMARK for Deletion")
    print(" [Q] or [ESC]   : Quit")
    print("================\n")

    current_idx = start_index
    
    # OpenCV Window Setup
    cv2.namedWindow("Dataset Auditor", cv2.WINDOW_NORMAL)
    
    while True:
        full_path, subset, filename = all_images[current_idx]
        
        img = cv2.imread(full_path)
        if img is None:
            print(f"Could not read {filename}, skipping.")
            current_idx = (current_idx + 1) % total_images
            continue
            
        h, w, _ = img.shape
        
        # --- Draw Labels ---
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        
        has_labels = False
        if os.path.exists(label_path):
            has_labels = True
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                try:
                    parts = list(map(float, line.split()))
                    class_id = int(parts[0])
                    x_center, y_center, box_w, box_h = parts[1:]
                    
                    # Convert YOLO normalized coordinates to absolute pixel coordinates
                    x1 = int((x_center - box_w/2) * w)
                    y1 = int((y_center - box_h/2) * h)
                    x2 = int((x_center + box_w/2) * w)
                    y2 = int((y_center + box_h/2) * h)
                    
                    color = COLORS.get(class_id, (255, 255, 255))
                    label_text = CLASS_NAMES.get(class_id, str(class_id))
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception as e:
                    print(f"Skipping bad box in {label_name}: {e}")

        # --- UI Overlay ---
        cv2.putText(img, f"Index: {current_idx + 1} / {total_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"Sensor: {subset}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"File: {filename}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Bottom: Controls Guide
        controls_help = "[X] Mark/Unmark   [C] Copy Name   [Space] Next   [A] Back"
        cv2.putText(img, controls_help, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Status: Marked or Not
        if filename in marked_files:
            cv2.rectangle(img, (0,0), (w, h), (0, 0, 255), 10)
            cv2.putText(img, "MARKED FOR DELETION", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        if not has_labels:
             cv2.putText(img, "NO LABEL FILE / NO OBJECTS", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Dataset Auditor", img)
        
        key = cv2.waitKey(0) & 0xFF
        
        # Navigation
        if key == ord('d') or key == ord(' '):
            current_idx += 1
            if current_idx >= total_images:
                print("End of dataset reached!")
                current_idx = total_images - 1
            save_progress_if_higher(current_idx, subset_choice)
            
        elif key == ord('a'):
            current_idx -= 1
            if current_idx < 0: current_idx = 0
            
        # Copy Filename
        elif key == ord('c'):
            pyperclip.copy(filename)
            print(f"Copied to clipboard: {filename}")
        
        # Action: Mark/Unmark
        elif key == ord('x'):
            if filename in marked_files:
                marked_files.remove(filename)
                print(f"Unmarked: {filename}")
            else:
                marked_files.add(filename)
                print(f"Marked: {filename}")
            save_marked_files(marked_files)
            
        # Quit
        elif key == ord('q') or key == 27: # 27 is ESC
            save_progress_if_higher(current_idx, subset_choice)
            break
            
    cv2.destroyAllWindows()
    print(f"Audit finished. Max progress saved for '{subset_choice}' sensor.")

if __name__ == "__main__":
    main()