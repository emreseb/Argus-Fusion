from ultralytics import YOLO
import cv2
import math
import os
import shutil
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


cap = cv2.VideoCapture("/home/emre/Desktop/NATO/BeforeSplit/DJI_20251028133726_0001_V.MP4")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="bestmodels/best13.pt",
    confidence_threshold=0.7,
    device="cuda"
)

classNames = ["Drone"]
work_dir_txt = "/home/emre/Desktop/NATO/coordinates"
work_dir_img = "/home/emre/Desktop/NATO/imgs"

if os.path.exists(work_dir_img):
    shutil.rmtree(work_dir_img)
if os.path.exists(work_dir_txt):
    shutil.rmtree(work_dir_txt)

os.makedirs(work_dir_img, exist_ok=True)
os.makedirs(work_dir_txt, exist_ok=True)

cv2.namedWindow('Yolo Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Yolo Detection', 1280, 720)

def save_img_and_coordinates(img, boxes_data, output_dir_img, output_dir_txt, frame_index):
    """Save frame once and corresponding bounding boxes in one txt file"""
    
    img_path = os.path.join(output_dir_img, f"frame_{frame_index}.jpg")
    cv2.imwrite(img_path, img)
    print(f"Saved frame to {img_path}")
    
    txt_path = os.path.join(output_dir_txt, f"frame_{frame_index}.txt")
    with open(txt_path, "w") as f:
        for box_data in boxes_data:
            cls, x1, y1, x2, y2, conf = box_data
            f.write(f"{cls} {x1} {y1} {x2} {y2}\n")
    print(f"Saved {len(boxes_data)} boxes to {txt_path}")

frame_counter = 0  

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame")
        break
    
    results = get_sliced_prediction(
        img,
        detection_model=detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    ) 

    object_prediction_list = results.object_prediction_list
    
    annotated_img = img.copy()
    boxes_data = []

    # Iterate through SAHI predictions
    for pred in object_prediction_list:
        # Extract bbox coordinates from SAHI ObjectPrediction
        bbox = pred.bbox  # This is a BoundingBox object
        x1 = int(bbox.minx)
        y1 = int(bbox.miny)
        x2 = int(bbox.maxx)
        y2 = int(bbox.maxy)
        
        # Extract class and confidence
        cls = pred.category.id
        confidence = pred.score.value
        
        # Print bbox coordinates
        print(f"Bbox: ({x1}, {y1}, {x2}, {y2}) | Class: {classNames[cls]} | Confidence: {confidence:.2f}")
        
        boxes_data.append((cls, x1, y1, x2, y2, confidence))
        
        # Draw on annotated image
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(annotated_img, f"{classNames[cls]} {confidence:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save frame and coordinates if detections exist
    if boxes_data:
        frame_counter += 1
        save_img_and_coordinates(img, boxes_data, work_dir_img, work_dir_txt, frame_counter)

    #cv2.imshow('Yolo Detection', annotated_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()