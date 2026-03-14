from ultralytics import YOLO

model = YOLO("bestmodels/best14.pt")


results = model.val()  # runs evaluation on the validation split
print(results.metrics)  # shows precision, recall, mAP50, mAP50-95
