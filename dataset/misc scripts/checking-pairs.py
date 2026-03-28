import os
from pathlib import Path

"""
This script checks if all txts have a img and vice versa. This step was important to 
ensure that we have only paired outputs.
"""


images_dir = "/home/emre/Desktop/NATO/DATASETv3/only_paired_output/img"
labels_dir = "/home/emre/Desktop/NATO/DATASETv3/only_paired_output/txt"

def check_dataset_pairs():
    print(f"Scanning images in: {images_dir}")
    print(f"Scanning labels in: {labels_dir}\n")

    image_files = list(Path(images_dir).rglob("*.jpg"))
    label_files = list(Path(labels_dir).rglob("*.txt"))

    image_names = {img.stem for img in image_files}
    label_names = {lbl.stem for lbl in label_files}

    images_without_labels = image_names - label_names
    labels_without_images = label_names - image_names

    print(f"Total Images Found: {len(image_names)}")
    print(f"Total Labels Found: {len(label_names)}")
    print("-" * 40)

    if not images_without_labels and not labels_without_images:
        print("Every image has a label, and every label has an image.")
    else:
        if images_without_labels:
            print(f"Found {len(images_without_labels)} images missing their .txt labels.")
            # Print the first 5 so you know exactly what to look for
            for name in list(images_without_labels)[:5]:
                print(f"   - Missing label for: {name}.jpg")
            if len(images_without_labels) > 5:
                print(f"   ... and {len(images_without_labels) - 5} more.")

        print()

        if labels_without_images:
            print(f"Found {len(labels_without_images)} 'Ghost' labels without a matching image.")
            for name in list(labels_without_images)[:5]:
                print(f"   - Orphaned label: {name}.txt")
            if len(labels_without_images) > 5:
                print(f"   ... and {len(labels_without_images) - 5} more.")

if __name__ == "__main__":
    check_dataset_pairs()
    