from ultralytics import YOLO
from pathlib import Path

model = YOLO("best11.pt") #training exisitng model

# This is the path to my YAML relative to my CWD
rel_path = "latest.yaml"

# This file _does_ exist
assert Path(rel_path).exists(), "File doesn't exist"

# This is fixed by using the _full_ path

full_path = Path(rel_path).resolve()

try:
    model.train(data=str(full_path), epochs=1, imgsz=640)
except RuntimeError:
    print("This doesn't print")