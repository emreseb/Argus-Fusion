import os
import re
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import math
from pathlib import Path

# --- Configuration ---
DATASET_ROOT = os.environ.get("DATASET_ROOT", "")
images_dir = os.environ.get(
    "DATASET_IMAGES_DIR",
    str(Path(DATASET_ROOT).expanduser() / "images") if DATASET_ROOT else "",
)
# Labels directory should point at the folder containing YOLO `.txt` files
labels_dir = os.environ.get(
    "DATASET_LABELS_DIR",
    str(Path(DATASET_ROOT).expanduser() / "labels") if DATASET_ROOT else "",
)
model_path = os.environ.get("MODEL_PATH", "best.pt")
classNames = ["bird", "drone"]

# --- Caching the Heavy Data & Models ---

@st.cache_resource
def load_sahi_model():
    """Loads the SAHI model only once to prevent lag when moving the slider."""
    print("Loading SAHI Model into memory...")
    device = os.environ.get("SAHI_DEVICE", "cpu")
    return AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=model_path,
        confidence_threshold=0.4,
        device=device,
    )

def extract_key_and_frame(filename):
    match = re.search(r'([A-Z]+\d+).*?(\d+)$', filename)
    if match: return match.group(1), int(match.group(2))
    return None, None

def get_sensor_type(root_path, filename):
    path_str = root_path.replace('\\', '/')
    if '/EO/' in path_str or path_str.endswith('/EO'): return 'EO'
    if '/IR/' in path_str or path_str.endswith('/IR'): return 'IR'
    if '_0_' in filename: return 'EO'
    if '_1_' in filename: return 'IR'
    return 'UNKNOWN'

@st.cache_data
def load_dataset():
    """Scans directories and pairs Images with their Ground Truth Labels."""
    if not images_dir or not labels_dir:
        return {}
    images_map = {}
    for root, _, files in os.walk(images_dir):
        for img_file in files:
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                name_no_ext = os.path.splitext(img_file)[0]
                key, frame = extract_key_and_frame(name_no_ext)
                sensor = get_sensor_type(root, name_no_ext)
                if key and frame is not None:
                    images_map[(sensor, key, frame)] = os.path.join(root, img_file)

    labels_map = {}
    for root, _, files in os.walk(labels_dir):
        for label_file in files:
            if label_file.endswith('.txt'):
                label_no_ext = os.path.splitext(label_file)[0]
                key, frame = extract_key_and_frame(label_no_ext)
                sensor = get_sensor_type(root, label_no_ext)
                if key and frame is not None:
                    labels_map[(sensor, key, frame)] = os.path.join(root, label_file)

    valid_pairs = {}
    for lock_key in images_map:
        valid_pairs[lock_key] = {
            "image": images_map[lock_key],
            "label": labels_map.get(lock_key, None) # Label might be missing, that's okay
        }
    return valid_pairs

# --- Drawing Logic ---
def draw_visuals(image_path, label_path, show_gt, show_sahi, sahi_model):
    """Draws Ground Truth (Red) and SAHI Predictions (Green) on the image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    # 1. Draw Ground Truth (RED)
    if show_gt and label_path and os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = parts[0]
                    x_c, y_c, w, h = map(float, parts[1:5])
                    
                    left = (x_c - w / 2) * img_width
                    top = (y_c - h / 2) * img_height
                    right = (x_c + w / 2) * img_width
                    bottom = (y_c + h / 2) * img_height
                    
                    draw.rectangle([left, top, right, bottom], outline="red", width=3)
                    draw.rectangle([left, top-15, left+30, top], fill="red")
                    draw.text((left+2, top-15), f"GT:{class_id}", fill="white")

    # 2. Draw SAHI Predictions (GREEN)
    if show_sahi and sahi_model:
        img_array = np.array(img)
        result = get_sliced_prediction(
            img_array,
            sahi_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=False
        )
        
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            confidence = math.ceil((obj.score.value * 100)) / 100
            cls = obj.category.id
            class_name = classNames[cls] if cls < len(classNames) else str(cls)
            
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
            draw.rectangle([x1, y1-15, x1+65, y1], fill="#00FF00")
            draw.text((x1+2, y1-15), f"{class_name} {confidence}", fill="black")
            
    return img

# --- Streamlit Web UI ---
st.set_page_config(layout="wide", page_title="SAHI Evaluation Viewer")

# Sidebar Controls
st.sidebar.title("⚙️ Controls")
show_gt = st.sidebar.checkbox("Show Ground Truth (Red)", value=True)
show_sahi = st.sidebar.checkbox("Run SAHI Inference (Green)", value=False)

if show_sahi:
    sahi_model = load_sahi_model()
    st.sidebar.success("✅ SAHI Model Loaded (MPS)")
else:
    sahi_model = None

st.title("🎯 YOLO + SAHI Dataset Evaluator")

dataset = load_dataset()

if not dataset:
    st.error(
        "No images found. Set `DATASET_ROOT` (or `DATASET_IMAGES_DIR` and "
        "`DATASET_LABELS_DIR`) to your dataset paths."
    )
else:
    # Group data for UI dropdowns
    sensors = sorted(list(set(k[0] for k in dataset.keys())))
    col1, col2, col3 = st.columns(3)
    
    with col1: selected_sensor = st.selectbox("Select Sensor", sensors)
    keys_for_sensor = sorted(list(set(k[1] for k in dataset.keys() if k[0] == selected_sensor)))
    
    with col2: selected_key = st.selectbox("Select Video/Key", keys_for_sensor)
    frames_for_key = sorted([k[2] for k in dataset.keys() if k[0] == selected_sensor and k[1] == selected_key])
    
    with col3:
        if frames_for_key:
            selected_frame = st.select_slider("Select Frame", options=frames_for_key)
        else:
            st.warning("No frames found.")
            selected_frame = None

    # Render Image
    if selected_frame is not None:
        lock = (selected_sensor, selected_key, selected_frame)
        paths = dataset[lock]
        
        # Draw and display
        with st.spinner("Processing Frame..."):
            annotated_img = draw_visuals(paths['image'], paths['label'], show_gt, show_sahi, sahi_model)
            st.image(annotated_img, use_container_width=True)