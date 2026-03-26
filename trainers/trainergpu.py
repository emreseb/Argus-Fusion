from ultralytics import YOLO
from pathlib import Path
import torch
#NEEDS CUDA
model = YOLO("bestmodels/best14.pt")

# This is the path to my YAML relative to my CWD
rel_path = "archive/drones.v1i.yolov11/data.yaml"

# This file _does_ exist
assert Path(rel_path).exists(), "File doesn't exist"

# This is fixed by using the _full_ path

full_path = Path(rel_path).resolve()
if torch.cuda.is_available()==True:
    try:

        model.train(data=str(full_path), 
                    epochs=70,
                    imgsz=640,
                    batch=-1,
                    device='cuda',
                    pretrained=False,
                    patience=50, #epochs to stop after plateu of performance
                    save=True,
                    fraction= 1, #% of training dataset to use
                    cache = True,
                    scale = 0.3,
                    optimizer = 'auto'
                    )
    except RuntimeError as e:
        print(f"Somehting went wrong :/. this specifically -> {e}")
    else:
        print("you broke your cuda AHAHAHAHAH")
    
results = model.val()  # runs evaluation on the validation split
print(results.metrics)  # shows precision, recall, mAP50, mAP50-95