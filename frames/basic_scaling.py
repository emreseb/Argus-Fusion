import cv2
import numpy as np

# Load images
img1 = cv2.imread('ExtractedFrames/vid1o/pic1o.png')  # image to align
img2 = cv2.imread('ExtractedFrames/vid2t/pic2t.png')  # reference image

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect SIFT keypoints and descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match features using FLANN matcher (better for SIFT)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Find matches using KNN (k=2 for ratio test)
matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Extract matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute homography using RANSAC
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp img1 to align with img2
h, w = img2.shape[:2]
aligned_img = cv2.warpPerspective(img1, M, (w, h))

# Save results
cv2.imwrite("aligned_img1.png", aligned_img)
cv2.imwrite("reference_img2.png", img2)

# Optional: visualize matches (for debugging)
matches_mask = mask.ravel().tolist()
draw_params = dict(matchColor=(0, 255, 0), 
                   singlePointColor=None, 
                   matchesMask=matches_mask, 
                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
cv2.imwrite("matches_debug.png", matched_img)

print(f"Alignment complete. Found {len(good_matches)} good matches.")
print("Output saved as 'aligned_img1.png'.")