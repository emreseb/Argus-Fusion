import cv2
import numpy as np
import time

# --- CONFIGURATION ---
# Path to the Matrix you just created
MATRIX_PATH = "homography_matrix.npy"

# Your video files (or camera streams)
IR_VIDEO_PATH = "BeforeSplit/demot.mp4"
OPT_VIDEO_PATH = "BeforeSplit/demoop.mp4"

# The resolution you used during your manual click test
# (This MUST match the manual_points.py resolution)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 1024

def main():
    # 1. Load the "Corrective Lens" (Homography Matrix)
    try:
        h_matrix = np.load(MATRIX_PATH)
        print("Loaded Homography Matrix successfully.")
    except FileNotFoundError:
        print("Error: homography_matrix.npy not found! Run the manual click script first.")
        return

    # 2. Open Video Streams
    cap_ir = cv2.VideoCapture(IR_VIDEO_PATH)
    cap_opt = cv2.VideoCapture(OPT_VIDEO_PATH)

    if not cap_ir.isOpened() or not cap_opt.isOpened():
        print("Error opening video files.")
        return

    print("Processing video... Press 'q' to stop.")

    while True:
        # Read a frame from both videos
        ret_ir, frame_ir = cap_ir.read()
        ret_opt, frame_opt = cap_opt.read()

        # Stop if either video ends
        if not ret_ir or not ret_opt:
            break

        # 3. PRE-PROCESS: Resize to the same size used in calibration
        # The matrix only works on the resolution it was calculated for!
        frame_ir = cv2.resize(frame_ir, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_opt = cv2.resize(frame_opt, (TARGET_WIDTH, TARGET_HEIGHT))

        # 4. APPLY THE MATRIX (The Magic Step)
        # This warps the IR frame to match the Optical frame perfectly
        aligned_ir = cv2.warpPerspective(frame_ir, h_matrix, (TARGET_WIDTH, TARGET_HEIGHT))

        # 5. VISUALIZE (Fusion)
        # Blend them 50/50 to see the alignment
        fusion_view = cv2.addWeighted(frame_opt, 0.5, aligned_ir, 0.5, 0)

        # Show the result
        cv2.imshow("Fused View", fusion_view)
        # Optional: Show side-by-side
        # combined = np.hstack((frame_opt, aligned_ir))
        # cv2.imshow("Side by Side (Fixed)", combined)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_ir.release()
    cap_opt.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()