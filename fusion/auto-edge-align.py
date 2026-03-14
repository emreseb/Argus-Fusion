import cv2
import numpy as np

# --- CONFIGURATION ---
SRC_IMAGE_PATH = 'ExtractedFrames/DJI_20251028130609_0004_T/DJI_20251028130609_0004_T_segment_1_frame_000.png' # IR
DST_IMAGE_PATH = 'ExtractedFrames/DJI_20251028130609_0004_V/DJI_20251028130609_0004_V_segment_1_frame_000.png' # Optical
MATRIX_OUTPUT = "homography_matrix.npy"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 1024

def load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load {path}")
        exit()
    return cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))

print("Loading images...")
src_img = load_and_resize(SRC_IMAGE_PATH)
dst_img = load_and_resize(DST_IMAGE_PATH)

# 1. PRE-PROCESS: Convert to Grayscale
src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

# 2. EDGE DETECTION (The Magic Step)
# We use Canny to turn the images into "line drawings"
# You might need to tweak these numbers (50, 150) depending on image contrast
print("Extracting Edges...")
src_edges = cv2.Canny(src_gray, 50, 150)
dst_edges = cv2.Canny(dst_gray, 50, 150)

# Show the user what the computer sees (The Skeletons)
cv2.imshow("IR Edges", src_edges)
cv2.imshow("Optical Edges", dst_edges)
cv2.waitKey(1000) # Pause for a second to let you see

# 3. FEATURE MATCHING ON EDGES
print("Detecting features on edge maps...")
sift = cv2.SIFT_create()

# Detect keypoints on the EDGE images, not the original images
kp1, des1 = sift.detectAndCompute(src_edges, None)
kp2, des2 = sift.detectAndCompute(dst_edges, None)

if des1 is None or des2 is None:
    print("Error: No features found in edges. Try adjusting Canny thresholds.")
    exit()

print(f"Found {len(kp1)} edge features in Source and {len(kp2)} in Destination.")

# Match features
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
matches = flann.knnMatch(des1, des2, k=2)

# Filter matches (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Good matches found: {len(good_matches)}")

if len(good_matches) > 10:
    # 4. CALCULATE HOMOGRAPHY
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC is crucial here to ignore noise in the edge detection
    h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    print("\nHomography calculated!")
    print(h_matrix)
    np.save(MATRIX_OUTPUT, h_matrix)

    # 5. VISUALIZE RESULT
    height, width, _ = dst_img.shape
    
    # Warp the ORIGINAL source image (not the edges) using the calculated matrix
    warped_img = cv2.warpPerspective(src_img, h_matrix, (width, height))
    
    # Blend
    overlay = cv2.addWeighted(dst_img, 0.5, warped_img, 0.5, 0)
    
    # Draw matches to help debug
    match_img = cv2.drawMatches(src_edges, kp1, dst_edges, kp2, good_matches[:20], None, flags=2)

    cv2.imshow("Edge Matches", match_img)
    cv2.imshow("Final Overlay", overlay)
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough matches found on edges.")