import cv2
import numpy as np

# --- CONFIGURATION ---
# Use the exact same images you used for the manual test
SRC_IMAGE_PATH = 'ExtractedFrames/DJI_20251028130609_0004_T/DJI_20251028130609_0004_T_segment_1_frame_000.png' # IR
DST_IMAGE_PATH = 'ExtractedFrames/DJI_20251028130609_0004_V/DJI_20251028130609_0004_V_segment_1_frame_000.png' # Optical
MATRIX_OUTPUT = "homography_matrix.npy"

# We MUST maintain the same working resolution to ensure the matrix fits your video later
TARGET_WIDTH = 1280
TARGET_HEIGHT = 1024

# Minimum number of matches required to calculate a valid homography
MIN_MATCH_COUNT = 20

def load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load {path}")
        exit()
    return cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))

print("Loading and resizing images...")
src_img = load_and_resize(SRC_IMAGE_PATH) # IR
dst_img = load_and_resize(DST_IMAGE_PATH) # Optical

# Convert to grayscale (SIFT works on intensity, not color)
src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

# --- 1. SIFT Feature Detection ---
print("Detecting SIFT keypoints...")
sift = cv2.SIFT_create()

# Find the keypoints and descriptors
kp1, des1 = sift.detectAndCompute(src_gray, None) # Source (IR)
kp2, des2 = sift.detectAndCompute(dst_gray, None) # Destination (Optical)

print(f"Found {len(kp1)} features in Source and {len(kp2)} in Destination.")

# --- 2. Feature Matching (FLANN) ---
# FLANN parameters (Fast Library for Approximate Nearest Neighbors)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# --- 3. Filter Matches (Lowe's Ratio Test) ---
# We only keep matches where the best match is significantly better than the 2nd best
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f"Good matches after filtering: {len(good_matches)}")

if len(good_matches) > MIN_MATCH_COUNT:
    # --- 4. Calculate Homography ---
    # Extract coordinates from the good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate Homography using RANSAC (robust to outliers)
    # RANSAC will ignore "bad" matches that don't fit the dominant geometric trend
    h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    print("\nHomography calculated successfully!")
    print(h_matrix)

    # Save the matrix
    np.save(MATRIX_OUTPUT, h_matrix)
    print(f"Matrix saved to {MATRIX_OUTPUT}")

    # --- 5. Visual Verification ---
    
    # Visual 1: Draw the actual matches
    matches_mask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0), # draw matches in green
                       singlePointColor=None,
                       matchesMask=matches_mask, # draw only inliers
                       flags=2)
    img_matches = cv2.drawMatches(src_img, kp1, dst_img, kp2, good_matches, None, **draw_params)
    cv2.imshow("SIFT Matches (Inliers)", img_matches)

    # Visual 2: The Warp Overlay (The "Truth" test)
    height, width, _ = dst_img.shape
    warped_img = cv2.warpPerspective(src_img, h_matrix, (width, height))
    
    # Create a 50/50 blend
    overlay = cv2.addWeighted(dst_img, 0.5, warped_img, 0.5, 0)
    cv2.imshow("Warped Overlay Check", overlay)

    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
    print("SIFT failed to find enough common features. This is common in IR-to-Optical.")
    print("Try adjusting the contrast of the IR image or fallback to manual points.")