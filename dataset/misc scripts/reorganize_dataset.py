import os
import shutil
from pathlib import Path

def reorganize_dataset(source_dir, output_images_dir, output_labels_dir):
    # Define common extensions (add or remove as needed for your specific dataset)
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    # Assuming YOLO or Pascal VOC formats. Change if your labels are different.
    label_exts = {'.txt', '.xml', '.json', '.csv'}

    # Create the target output directories if they don't already exist
    Path(output_images_dir).mkdir(parents=True, exist_ok=True)
    Path(output_labels_dir).mkdir(parents=True, exist_ok=True)

    source_path = Path(source_dir)
    
    # Counters to give you a summary at the end
    img_count = 0
    lbl_count = 0

    print(f"Scanning through: {source_path} ...")

    # .rglob('*') recursively searches all subfolders
    for file_path in source_path.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            
            # If the file is an image, copy to the images folder
            if ext in image_exts:
                dest = Path(output_images_dir) / file_path.name
                shutil.copy2(file_path, dest)
                img_count += 1
            
            # If the file is a label, copy to the labels folder
            elif ext in label_exts:
                dest = Path(output_labels_dir) / file_path.name
                shutil.copy2(file_path, dest)
                lbl_count += 1

    print("\n--- Reorganization Complete ---")
    print(f"Copied {img_count} images to: {output_images_dir}")
    print(f"Copied {lbl_count} labels to: {output_labels_dir}")

# --- Setup Paths ---
# Using the path visible in your screenshot
source_directory = r"D:\Teams 4th Year Spring semestar\Grad Project\DATASETv3 copy"

# Define where you want the new folders to be created
# (It's best to put them outside the source folder to avoid an infinite loop of copying)
new_images_folder = r"D:\Teams 4th Year Spring semestar\Grad Project\Reorganized_Set_V3\images"
new_labels_folder = r"D:\Teams 4th Year Spring semestar\Grad Project\Reorganized_Set_V3\labels"

# Run the function
reorganize_dataset(source_directory, new_images_folder, new_labels_folder)