import cv2
import numpy as np

# --- CONFIGURATION ---
SRC_IMAGE_PATH = 'ExtractedFrames/demot/demot_segment_2_frame_004.png'
DST_IMAGE_PATH = 'ExtractedFrames/demoop/demoop_segment_2_frame_004.png'

# Desired resolution for working/viewing
TARGET_WIDTH = 1280
TARGET_HEIGHT = 1024

# Lists to store the clicked points
src_points = []
dst_points = []

def select_points(event, x, y, flags, param):
    """Mouse callback function to record clicks."""
    img = param['img']
    pts_list = param['pts_list']
    win_name = param['win_name']

    if event == cv2.EVENT_LBUTTONDOWN:
        # Record the point
        pts_list.append((x, y))
        
        # Visual feedback: Draw a circle and coordinates
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1) 
        cv2.putText(img, f'{len(pts_list)}', (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow(win_name, img)

        print(f"Point recorded: {x}, {y}")

def get_points(image_path, point_storage, title):
    """Helper to open window and collect 4 points."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        exit()
    
    # --- FIX: RESIZE HERE (BEFORE showing the window) ---
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
        
    clone = img.copy()
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, select_points, {'img': clone, 'pts_list': point_storage, 'win_name': title})

    print(f"--- Please click 4 points on {title} ---")
    
    while True:
        cv2.imshow(title, clone)
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit early, or auto-break when 4 points are found
        if key == ord('q') or len(point_storage) == 4:
            break
            qqq
    cv2.destroyWindow(title)
    # Return the RESIZED image so we can use it for warping later
    return img 

# 1. Collect points from Source Image (IR)
# This will return the 1280x1024 version of the image
src_img_resized = get_points(SRC_IMAGE_PATH, src_points, "Source Image (To be warped)")

# 2. Collect points from Destination Image (Optical)
# This will return the 1280x1024 version of the image
dst_img_resized = get_points(DST_IMAGE_PATH, dst_points, "Destination Image (Reference)")

# 3. Check if we have enough points
if len(src_points) != 4 or len(dst_points) != 4:
    print("Error: You must select exactly 4 points on both images.")
else:
    print("Calculating Homography...")

    # Convert to NumPy arrays
    pts_src = np.array(src_points)
    pts_dst = np.array(dst_points)

    # Calculate Homography Matrix (H)
    h, status = cv2.findHomography(pts_src, pts_dst)
    
    # Warp the source image to align with destination
    # Use the dimensions of the RESIZED destination image
    height, width, channels = dst_img_resized.shape
    warped_image = cv2.warpPerspective(src_img_resized, h, (width, height))

    # Overlay for visual check (50% transparency)
    overlay = cv2.addWeighted(dst_img_resized, 0.5, warped_image, 0.5, 0)

    cv2.imshow("Warped Result", warped_image)
    cv2.imshow("Overlay (Check Alignment)", overlay)
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optional: Save the matrix for later use
    np.save("homography_matrix.npy", h)
    print("Matrix saved to homography_matrix.npy")