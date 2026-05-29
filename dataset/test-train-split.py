import shutil
import random
from pathlib import Path

def split_yolo_dataset(base_dir: Path, output_dir: Path, train_ratio=0.8, val_ratio=0.1):
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels" 
    
    image_paths = list(images_dir.glob("*.jpg"))
    
    random.seed(42)
    random.shuffle(image_paths)
    
    # Calculate the split indices
    total_images = len(image_paths)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Slice the list into the three groups
    splits = {
        "train": image_paths[:train_end],
        "val": image_paths[train_end:val_end],
        "test": image_paths[val_end:]
    }
    
    print(f"Total images found: {total_images}")
    
    for split_name, paths in splits.items():
        print(f"Copying {len(paths)} files to {split_name}...")
        
        split_img_dir = output_dir / "images" / split_name
        split_lbl_dir = output_dir / "labels" / split_name
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in paths:
            # 1. Copy the image
            shutil.copy(img_path, split_img_dir)
            
            txt_filename = img_path.stem + ".txt"
            txt_path = labels_dir / txt_filename
            
            # 3. Copy the label if it exists
            if txt_path.exists():
                shutil.copy(txt_path, split_lbl_dir)
            else:
                print(f"⚠️ Warning: No matching label found for {img_path.name}")

def main():
    base_dir = Path("/home/emre/Desktop/NATO/DATASETv3/training-eo-ir-seperated/IR")
    
    output_dir = Path("/home/emre/Desktop/NATO/DATASETv3/training-eo-ir-seperated/IR-train-test")
    
    if not base_dir.exists():
        print("Error: Base directory not found.")
        return
        
    split_yolo_dataset(base_dir, output_dir)
    print("\n✅ Dataset successfully split into Train/Val/Test!")

if __name__ == "__main__":
    main()