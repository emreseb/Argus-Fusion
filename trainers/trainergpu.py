from ultralytics import YOLO
from pathlib import Path
import torch
#NEEDS CUDA
model = YOLO("yolo11m.pt")

# This is the path to my YAML relative to my CWD
rel_path = "/users/home/aiproject/ProjectAi4Cuav/REPO/DATASET-TOOLS/overfit_dataset.yaml"

# This file _does_ exist
assert Path(rel_path).exists(), "File doesn't exist"

# This is fixed by using the _full_ path

full_path = Path(rel_path).resolve()
if torch.cuda.is_available()==True:
    try:

        model.train(
        data=str(full_path), 
        epochs=300,           # Set to 300 as requested
        imgsz=1024,          # INCREASED: Drone objects are small; 640 often loses them. L40 can handle 1024.
        batch=32,            # FIXED: -1 is okay, but 32 is a "sweet spot" for L40 with 1024px images.
        device=0,            # Use index 0 for a single GPU
        patience=50,         # Keeps your "early stop" safety net
        save=True,
        cache='ram',         # L40 systems usually have high RAM; 'ram' is faster than 'disk'
        optimizer='AdamW',   # Better for fine-tuning and modern YOLO versions than 'auto'
        lr0=0.01,            # Standard starting point
        cos_lr=True,         # Use Cosine Learning Rate scheduler (better for 300 epochs)
        mosaic=1.0,          # CRITICAL for drone data (combines 4 images to help with small objects)
        mixup=0.1,           # Adds some robustness to occlusion
        amp=True             # Automatic Mixed Precision (faster, uses less VRAM)
)
    except RuntimeError as e:
        print(f"Somehting went wrong :/. this specifically -> {e}")
    else:
        print("you broke your cuda AHAHAHAHAH")
    
results = model.val()  # runs evaluation on the validation split
print(results.metrics)  # shows precision, recall, mAP50, mAP50-95


 
