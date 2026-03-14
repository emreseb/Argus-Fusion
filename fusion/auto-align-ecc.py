import cv2
import numpy as np

# --- CONFIGURATION ---
SRC_IMAGE_PATH = 'ExtractedFrames/demot/demot_segment_2_frame_004.png' # IR
DST_IMAGE_PATH = 'ExtractedFrames/demoop/demoop_segment_2_frame_004.png' # Optical
MATRIX_OUTPUT = "homography_matrix.npy"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 1024

def load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load {path}")
        exit()
    return cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))

print("Loading and resizing images...")
src_img = load_and_resize(SRC_IMAGE_PATH) # IR (To be warped)
dst_img = load_and_resize(DST_IMAGE_PATH) # Optical (Reference)

# Convert to grayscale (ECC requires grayscale)
src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

# --- ECC ALGORITHM ---
print("Running ECC Algorithm (this might take a few seconds)...")

# 1. Define the motion model (Homography is the most flexible)
warp_mode = cv2.MOTION_HOMOGRAPHY

# 2. Initialize the matrix to Identity (start with no change)
if warp_mode == cv2.MOTION_HOMOGRAPHY:
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else:
    warp_matrix = np.eye(2, 3, dtype=np.float32)

# 3. Define termination criteria
# Stop if the algorithm runs for 5000 iterations OR if the correlation improves by less than 1e-10
number_of_iterations = 5000
termination_eps = 1e-10
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

try:
    # 4. Run the ECC algorithm
    # inputMask is None (we use the whole image)
    # gaussFiltSize is 5 (smooths image slightly to prevent getting stuck in noise)
    cc, warp_matrix = cv2.findTransformECC(src_gray, dst_gray, warp_matrix, warp_mode, criteria, None, 5)

    print(f"\nConvergence achieved! Correlation: {cc:.4f}")
    print("Calculated Homography Matrix:")
    print(warp_matrix)

    # Save the matrix
    np.save(MATRIX_OUTPUT, warp_matrix)
    print(f"Matrix saved to {MATRIX_OUTPUT}")

    # --- VISUALIZATION ---
    height, width, _ = dst_img.shape
    
    # Warp the source image
    aligned_image = cv2.warpPerspective(src_img, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Create a Blend
    overlay = cv2.addWeighted(dst_img, 0.5, aligned_image, 0.5, 0)

    cv2.imshow("ECC Alignment Result", aligned_image)
    cv2.imshow("Overlay Check", overlay)
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except cv2.error as e:
    print("\n--- ECC FAILED ---")
    print("ECC could not converge. This usually means the images are too different or too far apart initially.")
    print("Detailed error:", e)