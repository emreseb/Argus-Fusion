import cv2
import math
import os
import glob
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- 1. Configuration ---
images_dir = "/Users/emre/Desktop/DATASETv3/images"
model_path = "best.pt"
classNames = ["drone"] 

# --- 2. Load SAHI Model ---
print("Loading SAHI Model (Ultralytics Backend)...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=model_path,
    confidence_threshold=0.4,
    device="mps" # Forces Apple Silicon GPU
)

# --- 3. Load and Sort All Image Paths ---
print(f"Scanning for images in: {images_dir}")
search_pattern = os.path.join(images_dir, "**", "*.jpg")
image_paths = sorted(glob.glob(search_pattern, recursive=True))

if not image_paths:
    print("Error: No .jpg images found in the directory!")
    exit()

print(f"Found {len(image_paths)} images.")
print("Controls: [A] = Previous | [D] = Next | [Q] = Quit")

# --- 4. Setup Window ---
cv2.namedWindow('SAHI + YOLO Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SAHI + YOLO Detection', 1280, 720)

# --- 5. Main Interactive Loop ---
current_index = 0

while True:
    img_path = image_paths[current_index]
    img = cv2.imread(img_path)
    
    if img is not None:
        # SAHI requires RGB, but OpenCV loads in BGR. We must convert it first.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run SAHI Sliced Inference
        # This chops the image into 640x640 squares with a 20% overlap so drones aren't cut in half
        result = get_sliced_prediction(
            img_rgb,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=False
        )

        # Draw Boxes from SAHI Results
        for object_prediction in result.object_prediction_list:
            # Extract coordinates
            bbox = object_prediction.bbox
            x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
            
            # Extract score and class ID
            confidence = math.ceil((object_prediction.score.value * 100)) / 100
            cls = object_prediction.category.id
            
            # Draw Rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Prevent crashing if the model detects an unknown class
            class_name = classNames[cls] if cls < len(classNames) else str(cls)
            
            # Draw Text
            org = [x1, max(20, y1 - 10)] 
            cv2.putText(img, f"{class_name} {confidence}", org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # --- UI Overlay ---
        filename = os.path.basename(img_path)
        progress_text = f"Image {current_index + 1} of {len(image_paths)} | {filename} | SAHI Active"
        
        cv2.putText(img, progress_text, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(img, progress_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    else:
        print(f"⚠️ Warning: Could not read {img_path}.")

    cv2.imshow('SAHI + YOLO Detection', img)

    # --- Keyboard Listener ---
    key = cv2.waitKey(0) & 0xFF

    if key == ord('a') or key == ord('A'):
        current_index = max(0, current_index - 1)
    elif key == ord('d') or key == ord('D'):
        current_index = min(len(image_paths) - 1, current_index + 1)
    elif key == ord('q') or key == ord('Q'):
        print("Exiting viewer...")
        break

cv2.destroyAllWindows()