from ultralytics import YOLO

# Load your best weights
model = YOLO('/home/emre/Desktop/NATO/code/runs/detect/train19/weights/best.pt')

# Validate the model
metrics = model.val(data='/home/emre/Desktop/NATO/code/yaml-files/ir-model.yaml', split='val')

# Print out the exact mAP50-95 score
print(f"Final mAP50-95: {metrics.box.map}")
print(f"Final mAP50: {metrics.box.map50}")
print(f"Final mAP75: {metrics.box.map75}")
