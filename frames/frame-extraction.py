#!/usr/bin/env python3
import os
import argparse
import time
import hashlib
import cv2
from moviepy.editor import VideoFileClip

def get_video_hash(video_path):
    """Generate a hash for video identification based on file name and size"""
    file_stats = os.stat(video_path)
    file_size = file_stats.st_size
    base_name = os.path.basename(video_path)
    hash_input = f"{base_name}-{file_size}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def is_already_processed(video_path, processed_videos_file):
    """Check if a video has already been processed"""
    video_hash = get_video_hash(video_path)
    
    if not os.path.exists(processed_videos_file):
        return False
        
    with open(processed_videos_file, 'r') as f:
        processed_hashes = f.read().splitlines()
        
    return video_hash in processed_hashes

def mark_as_processed(video_path, processed_videos_file):
    """Mark a video as processed by recording its hash"""
    video_hash = get_video_hash(video_path)
    
    with open(processed_videos_file, 'a+') as f:
        f.write(f"{video_hash}\n")

def extract_one_frame_per_x(video_path, output_dir, segment_name):
    """Extract one frame per given x time from a video segment"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count 

    count = 0
    while count <= duration:
        frame_no = int(count * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success, frame = cap.read()

        if not success:
            break

        frame_filename = f"{segment_name}_frame_{int(count):03d}.png"
        output_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(output_path, frame)
        print(f"Saved frame: {output_path}")

        count += 2  # Extract one frame every {count} seconds 

    cap.release()

def extract_all_frames(video_path, video_frames_dir, frame_start_index):
    """Extract all frames from a video segment into the video's folder"""
    cap = cv2.VideoCapture(video_path)
    frame_index = frame_start_index

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_filename = f"frame_{frame_index:05d}.png"
        output_path = os.path.join(video_frames_dir, frame_filename)
        cv2.imwrite(output_path, frame)
        print(f"Saved frame: {output_path}")

        frame_index += 1

    cap.release()
    return frame_index  


def split_video(video_path, output_dir, frame_output_dir, segment_duration=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)
    
    # Create folder for this video's frames
    video_frames_dir = os.path.join(frame_output_dir, name)
    os.makedirs(video_frames_dir, exist_ok=True)

    frame_index = 0  

    try:
        clip = VideoFileClip(video_path)
        duration = int(clip.duration)
        segments = duration // segment_duration

        for i in range(segments + 1):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)

            if end_time <= start_time:
                continue

            segment = clip.subclip(start_time, end_time)

            segment_filename = f"{name}_segment_{i+1:03d}.mp4"
            segment_path = os.path.join(output_dir, segment_filename)

            segment.write_videofile(segment_path,
                                    codec="libx264",
                                    audio_codec="aac",
                                    temp_audiofile=f"temp-audio-{i}.m4a",
                                    remove_temp=True)

            print(f"Created segment {i+1}/{segments+1}: {segment_path}")

            # Extract all frames from this segment and store in video-specific folder
            frame_index = extract_one_frame_per_x(segment_path, video_frames_dir, f"{name}_segment_{i+1}")

        clip.close()

        print(f"Video split complete. {segments+1} segments created in '{output_dir}'")
        return True
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return False


def process_videos(input_dir, output_dir, frame_output_dir, segment_duration, processed_videos_file):
    """
    Process all videos in input directory AND subdirectories recursively.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(frame_output_dir):
        os.makedirs(frame_output_dir)
        
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    # os.walk traverses the directory tree
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check extension
            if os.path.splitext(file)[1].lower() in video_extensions:
                video_path = os.path.join(root, file)
                
                # --- BUG FIX: Check if already processed ---
                if is_already_processed(video_path, processed_videos_file):
                    print(f"Skipping {file} - Already processed.")
                    continue
                # -------------------------------------------

                print(f"Processing {video_path}...")
                
                # Pass the full video path to the split function
                success = split_video(video_path, output_dir, frame_output_dir, segment_duration)
                
                if success:
                    mark_as_processed(video_path, processed_videos_file)

