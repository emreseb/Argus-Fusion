"""
Time based frame extractor for dual-sensor video datasets.
"""
#!/usr/bin/env python3
import cv2
import os
import argparse
import time
import hashlib
import sys
exp_type = "M"

# --- CONFIGURATION ---
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
PROCESSED_LOG_FILE = ".processed_log"

def get_video_hash(video_path):
    """Generate a hash based on file size and name to track processed files."""
    try:
        file_stats = os.stat(video_path)
        file_size = file_stats.st_size
        base_name = os.path.basename(video_path)
        hash_input = f"{base_name}-{file_size}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    except FileNotFoundError:
        return None

def load_processed_hashes():
    """Load the list of already processed video hashes."""
    if not os.path.exists(PROCESSED_LOG_FILE):
        return set()
    with open(PROCESSED_LOG_FILE, 'r') as f:
        return set(line.strip() for line in f)

def save_processed_hash(video_hash):
    """Mark a video as processed."""
    with open(PROCESSED_LOG_FILE, 'a') as f:
        f.write(f"{video_hash}\n")

def get_common_id(filename):
    """
    Extracts the unique ID starting from 'E' in the filename.
    Example: '1_110_0_E19.mp4' -> 'E19.mp4'
    """
    # Look for the identifier 'E'
    if exp_type in filename:
        # Find the index where 'E' starts
        # We use rfind to find the last 'E' if there are multiple, 
        # or generally find the specific event ID.
        # Given '1_110_0_E19', we want the part starting at E.
        
        # Strategy: Split by underscore and find the part starting with E
        parts = filename.split('_')
        for part in parts:
            if part.startswith(exp_type):
                # We return this part plus the extension if it was split off,
                # but simpler is to just slice the string from 'E' onwards.
                index = filename.find(part)
                return filename[index:]
        
        # Fallback: Simple slice from the first 'E' found
        index = filename.find(exp_type)
        return filename[index:]
        
    # If no 'E' is found, return the original filename
    return filename

def extract_synced_frames(path1, path2, output_dir, target_fps):
    """
    Syncs two videos and extracts pairs at a specific frame rate.
    """
    raw_filename_s1 = os.path.basename(path1)
    raw_filename_s2 = os.path.basename(path2)
    
    # Use the Common ID (the E-number) for the folder name
    # Remove extension from the ID to get the folder name (e.g. E19)
    common_name = os.path.splitext(get_common_id(raw_filename_s1))[0]
    
    # Extract the full parent video name without extension for the frames
    parent_name_s1 = os.path.splitext(raw_filename_s1)[0]
    parent_name_s2 = os.path.splitext(raw_filename_s2)[0]
    
    
    # Create output subdirectories
    out_s1 = os.path.join(output_dir, "IR", common_name)
    out_s2 = os.path.join(output_dir, "EO", common_name)
    os.makedirs(out_s1, exist_ok=True)
    os.makedirs(out_s2, exist_ok=True)

    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print(f"Error opening video files: {path1} or {path2}")
        return False

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    
    # --- FPS CALCULATION ---
    # Calculate skip interval based on target FPS
    if target_fps and target_fps > 0:
        frame_interval = int(fps1 / target_fps)
        if frame_interval < 1: frame_interval = 1
    else:
        frame_interval = 1
    # -----------------------

    print(f"Syncing: {common_name}")
    print(f"  > Sensor 1: {fps1} FPS | Sensor 2: {fps2} FPS")
    print(f"  > Target Extraction: {target_fps} FPS (Every {frame_interval}th frame)")

    frame_count = 0
    saved_count = 0
    
    while True:
        # 1. Read Master Frame (Sensor 1)
        ret1, frame1 = cap1.read()
        if not ret1:
            break

        # Check if we should save this frame based on interval
        if frame_count % frame_interval == 0:
            
            # 2. Calculate EXACT timestamp of this frame
            current_time_ms = (frame_count / fps1) * 1000

            # 3. Force Sensor 2 to jump to that timestamp
            cap2.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
            ret2, frame2 = cap2.read()

            if not ret2:
                break # Sensor 2 finished early

            # 4. Save the pair using the full parent video name
            img_name_s1 = f"{parent_name_s1}_frame{saved_count:06d}.jpg"
            img_name_s2 = f"{parent_name_s2}_frame{saved_count:06d}.jpg"
            
            cv2.imwrite(os.path.join(out_s1, img_name_s1), frame1)
            cv2.imwrite(os.path.join(out_s2, img_name_s2), frame2)

            saved_count += 1
            
            if saved_count % 50 == 0:
                sys.stdout.write(f"\r  > Extracted {saved_count} pairs...")
                sys.stdout.flush()

        frame_count += 1

    print(f"\n  > Finished. Total pairs: {saved_count}")
    cap1.release()
    cap2.release()
    return True

def scan_and_process(dir1, dir2, output_dir, processed_hashes, target_fps):
    """
    Scans directories for matching Common IDs and processes them.
    """
    
    # 1. Index Sensor 2 first
    # Create a map: { common_id : full_path }
    sensor2_map = {}
    if os.path.exists(dir2):
        for root, _, files in os.walk(dir2):
            for file in files:
                if os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS:
                    # Get ID (e.g., E19.mp4)
                    c_id = get_common_id(file)
                    sensor2_map[c_id] = os.path.join(root, file)
    
    # 2. Scan Sensor 1 and look for matches in the map
    for root, _, files in os.walk(dir1):
        for file in files:
            if os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS:
                path1 = os.path.join(root, file)
                
                # Get ID (e.g., E19.mp4)
                c_id = get_common_id(file)
                
                # Check if we have a match in Sensor 2
                if c_id not in sensor2_map:
                    # Optional: Print warning only if needed
                    # print(f"Waiting for match for: {file} (ID: {c_id})") 
                    continue

                path2 = sensor2_map[c_id]

                # Check if already processed
                v_hash = get_video_hash(path1)
                if v_hash in processed_hashes:
                    continue

                # PROCESS THE PAIR
                print(f"Found new pair: {file} + {os.path.basename(path2)}")
                success = extract_synced_frames(path1, path2, output_dir, target_fps)
                
                if success:
                    save_processed_hash(v_hash)
                    processed_hashes.add(v_hash)

def main():
    parser = argparse.ArgumentParser(description="Synchronize and extract frames from dual-sensor video setups.")
    
    parser.add_argument("--sensor1-dir", 
                        help="Input directory for Sensor 1 videos (Master time reference)",
                        default="/media/emre/SEB/DATASET-(vid)/18.12.25/IR")
    parser.add_argument("--sensor2-dir",
                        help="Input directory for Sensor 2 videos (Slave time reference)",
                        default="/media/emre/SEB/DATASET-(vid)/18.12.25/EO")
    parser.add_argument("-o", "--output-dir", default="datasetf_output", 
                        help="Directory to save extracted frame pairs")
    parser.add_argument("-w", "--watch", action="store_true",
                        help="Keep running and watch folders for new files")
    parser.add_argument("-t", "--interval", type=int, default=10,
                        help="Seconds to wait between checks in watch mode")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="How many frames to extract per second (default: 1.0)")

    args = parser.parse_args()

    # Ensure directories exist
    if not os.path.exists(args.sensor1_dir) or not os.path.exists(args.sensor2_dir):
        print("Error: Input directories do not exist.")
        return

    processed_hashes = load_processed_hashes()

    print("--- Video Sync & Extract Tool Started ---")
    print(f"Sensor 1: {args.sensor1_dir}")
    print(f"Sensor 2: {args.sensor2_dir}")
    print(f"Extraction Rate: {args.fps} frames/sec")

    if args.watch:
        print(f"Watching for new videos (Check interval: {args.interval}s)...")
        try:
            while True:
                scan_and_process(args.sensor1_dir, args.sensor2_dir, args.output_dir, processed_hashes, args.fps)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nWatch mode stopped.")
    else:
        scan_and_process(args.sensor1_dir, args.sensor2_dir, args.output_dir, processed_hashes, args.fps)
        print("Batch processing complete.")

if __name__ == "__main__":
    main()