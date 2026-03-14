from ultralytics import YOLO
import torch


model = YOLO("yolo11x.pt") 

# 2. Train with M4-specific settings
results = model.train(
    data="your_dataset.yaml", 
    epochs=300,            # High epochs to ensure overfitting
    imgsz=640,             # Standard resolution (or 1280 for tiny objects)
    device="mps",          # CRITICAL: Forces use of Mac GPU
    batch=16,              # Adjust based on your RAM (M4 handles 16-32 easily)
    workers=0,             # Helps stability on macOS
    amp=False,             # Disabling Automatic Mixed Precision can prevent NaNs on MPS
    cache=True             # Uses M4's fast Unified Memory to speed up training
)