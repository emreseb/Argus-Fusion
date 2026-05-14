import cv2
import numpy as np


def predict_day_night(image_path, threshold=140):

    # Load image
    image_bgr = cv2.imread(image_path)

    # Safety check
    if image_bgr is None:
        return "IMAGE NOT FOUND"

    # Convert BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize
    image_rgb = cv2.resize(image_rgb, (1100, 600))

    # Convert RGB -> HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Calculate average brightness
    avg_brightness = np.mean(hsv[:, :, 2])

    # Predict
    if avg_brightness > threshold:
        return "DAY"
    else:
        return "NIGHT"
