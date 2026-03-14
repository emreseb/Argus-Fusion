import cv2
import numpy as np

# --- load
rgb = cv2.imread("DJI_20251001123904_0002_V.JPG")            # high-res color
thermal = cv2.imread("DJI_20251001123904_0002_T.JPG")    # low-res thermal (e.g. 640x512)

# --- resize thermal to rgb canvas scale (preserve aspect by simple scale)
# Option A: scale thermal to exactly rgb size (easy)
thermal_up = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

# If you know sensor FOVs and want to preserve pixel scale, compute scale factor instead:
# scale_x = rgb.shape[1] / thermal.shape[1]
# scale_y = rgb.shape[0] / thermal.shape[0]
# thermal_up = cv2.resize(thermal, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)

# --- preprocess: grayscale + optional equalization to help cross-modality matching
gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
gray_th = cv2.cvtColor(thermal_up, cv2.COLOR_BGR2GRAY)
gray_th = cv2.equalizeHist(gray_th)    # helps contrast for thermal

# --- feature detection (SIFT recommended)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_rgb, None)
kp2, des2 = sift.detectAndCompute(gray_th, None)

# --- match descriptors (FLANN or BF + ratio test)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# ratio test
good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print("Good matches:", len(good))

if len(good) < 8:
    raise RuntimeError("Too few good matches for reliable homography. Try different preprocessing or detectors.")

# --- build matched points
pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)  # rgb points
pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)  # thermal points

# --- compute homography with RANSAC
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
inliers = mask.sum()
print("Inliers:", int(inliers), " / ", len(good))

if H is None:
    raise RuntimeError("Homography estimation failed. Try affine or other methods.")

# --- warp thermal to rgb coordinate frame
aligned_thermal = cv2.warpPerspective(thermal_up, H, (rgb.shape[1], rgb.shape[0]), flags=cv2.INTER_LINEAR)

cv2.imwrite("aligned_thermal.jpg", aligned_thermal)

# --- evaluation: reprojection error (mean distance between matched keypoints after warping)
pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1,3).T   # 3 x N
proj = (H @ pts2_h).T
proj = proj[:, :2] / proj[:, 2:3]
dists = np.linalg.norm(proj - pts1.reshape(-1,2), axis=1)
print("Mean reprojection error (px):", dists.mean(), " Median:", np.median(dists))

# --- easy visual overlay for inspection
blend = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), 0.6, cv2.cvtColor(aligned_thermal, cv2.COLOR_BGR2GRAY), 0.4, 0)
cv2.imwrite("overlay_gray.jpg", blend)
