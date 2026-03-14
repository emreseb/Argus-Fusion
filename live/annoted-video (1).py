import cv2
import os
from ultralytics import YOLO
import numpy as np

# --- Configuration ---
MODEL_PATH = "yolo12x.pt"  # Can be COCO model (e.g., yolov8x.pt) or custom
VIDEO_PATH = "birds.mp4"
OUTPUT_PATH = "output_annotated_video.mp4"
FRAME_SKIP = 5
CONF_THRESHOLD = 0.4
PROCESSING_WIDTH = 1280

# We will detect ALL classes, then relabel in post-processing
CLASSES_TO_DETECT = None  # None = detect all  NEVER USED???


def relabel_as_drone_or_bird(results, model_names):
    """
    Modifies results in-place: 
    - Keeps 'bird' as 'bird'
    - Changes ALL other detected classes to 'drone'
    """
    if not results or len(results[0].boxes) == 0:
        return results

    # Get boxes, classes, confidences
    boxes = results[0].boxes
    cls = boxes.cls.cpu().numpy()  # class indices
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    # Prepare new labels
    new_labels = []
    new_cls = []
    for i, class_id in enumerate(cls):
        class_name = model_names[int(class_id)].lower()
        if class_name == "bird":
            new_labels.append(f"bird {conf[i]:.2f}")
            new_cls.append(1)  # arbitrary: bird = 1
        else:
            new_labels.append(f"drone {conf[i]:.2f}")
            new_cls.append(0)  # drone = 0

    # We'll draw manually since .plot() uses original labels
    # So return frame + custom annotations
    return xyxy, new_labels, new_cls


def draw_custom_boxes(frame, xyxy_list, labels):
    """Draw bounding boxes and custom labels on frame."""
    frame_copy = frame.copy()
    for (xyxy, label) in zip(xyxy_list, labels):
        x1, y1, x2, y2 = map(int, xyxy)
        # Draw box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.6, 2)[0]
        cv2.rectangle(frame_copy, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
        # Draw label text
        cv2.putText(frame_copy, label, (x1, y1 - 5), font, 0.6, (0, 0, 0), 2)
    return frame_copy


def process_video(model_path, video_path, output_path, frame_skip, conf):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("Model classes:", model.names)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    aspect = orig_h / orig_w
    proc_h = int(PROCESSING_WIDTH * aspect)
    proc_h = proc_h if proc_h % 2 == 0 else proc_h - 1
    proc_size = (PROCESSING_WIDTH, proc_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, proc_size)

    frame_count = 0
    last_annotated_frame = None

    print("🚀 Processing video with 'not bird = drone' logic...")
    #????????????????????????????????*

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, proc_size, interpolation=cv2.INTER_AREA)

        if frame_count % frame_skip == 0:
            # Detect ALL objects
            results = model(frame_resized, conf=conf, verbose=False)
            
            # Apply relabeling logic
            if results and len(results[0].boxes) > 0:
                xyxy, labels, _ = relabel_as_drone_or_bird(results, model.names)
                last_annotated_frame = draw_custom_boxes(frame_resized, xyxy, labels)
            else:
                last_annotated_frame = frame_resized.copy()
        else:
            # Reuse last annotated frame if available, else just resized frame
            if last_annotated_frame is None:
                last_annotated_frame = frame_resized.copy()

        out.write(last_annotated_frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"   Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("\n✅ Done!")
    print(f"📁 Output: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    try:
        process_video(
            model_path=MODEL_PATH,
            video_path=VIDEO_PATH,
            output_path=OUTPUT_PATH,
            frame_skip=FRAME_SKIP,
            conf=CONF_THRESHOLD
        )
    except Exception as e:
        print(f"❌ Error: {e}")