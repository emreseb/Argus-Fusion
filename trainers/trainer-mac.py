from ultralytics import YOLO
import torch

torch.mps.empty_cache()

# 1. THE FIX: Switch from 'x' (Extra Large) to 'n' (Nano) or 's' (Small)
model = YOLO("yolo11n.pt") 

results = model.train(
    data="overfit_dataset.yaml", 
    epochs=300,            
    imgsz=640,             
    device="mps",              
    batch=32,              
    workers=4,             
    cache=False             
)