import os
import re
import streamlit as st
from PIL import Image, ImageDraw

# --- Configuration ---
# Hardcoded to your exact paths
images_dir = "/Users/emre/Desktop/DATASETv3/images" 
labels_dir = "/Users/emre/Desktop/DATASETv3/labels/obj_train_data" 

# --- Exact Mapping Logic from Previous Script ---
def extract_key_and_frame(filename):
    match = re.search(r'([A-Z]+\d+).*?(\d+)$', filename)
    if match:
        return match.group(1), int(match.group(2))
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
    """Scans directories and returns a dictionary of valid Image/Label pairs."""
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

    # Find the matching pairs
    valid_pairs = {}
    for lock_key in images_map:
        if lock_key in labels_map:
            valid_pairs[lock_key] = {
                "image": images_map[lock_key],
                "label": labels_map[lock_key]
            }
    return valid_pairs

def draw_yolo_boxes(image_path, label_path):
    """Opens image, reads YOLO txt, and draws the bounding boxes."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO Format: class x_center y_center width height (Normalized 0-1)
                class_id = parts[0]
                x_c, y_c, w, h = map(float, parts[1:5])
                
                # Convert normalized coordinates to absolute pixels
                left = (x_c - w / 2) * img_width
                top = (y_c - h / 2) * img_height
                right = (x_c + w / 2) * img_width
                bottom = (y_c + h / 2) * img_height
                
                # Draw Box
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                # Draw Label Background and Text
                draw.rectangle([left, top-15, left+20, top], fill="red")
                draw.text((left+2, top-15), str(class_id), fill="white")
                
    return img

# --- Streamlit Web UI ---
st.set_page_config(layout="wide", page_title="Dataset Viewer")
st.title("🎯 YOLO Dataset Viewer")

dataset = load_dataset()

if not dataset:
    st.error("No matching image/label pairs found. Check your paths!")
else:
    # 1. Group data for UI dropdowns
    sensors = sorted(list(set(k[0] for k in dataset.keys())))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_sensor = st.selectbox("Select Sensor", sensors)
        
    keys_for_sensor = sorted(list(set(k[1] for k in dataset.keys() if k[0] == selected_sensor)))
    
    with col2:
        selected_key = st.selectbox("Select Video/Key", keys_for_sensor)
        
    frames_for_key = sorted([k[2] for k in dataset.keys() if k[0] == selected_sensor and k[1] == selected_key])
    
    with col3:
        if frames_for_key:
            selected_frame = st.select_slider("Select Frame", options=frames_for_key)
        else:
            st.warning("No frames found.")
            selected_frame = None

    # 2. Render the selected image
    if selected_frame is not None:
        lock = (selected_sensor, selected_key, selected_frame)
        paths = dataset[lock]
        
        st.write(f"**Image:** `{os.path.basename(paths['image'])}`")
        st.write(f"**Label:** `{os.path.basename(paths['label'])}`")
        
        # Draw and display
        annotated_img = draw_yolo_boxes(paths['image'], paths['label'])
        st.image(annotated_img, use_container_width=True)