import os
import glob
import shutil

# The folder where everything currently is, and where you want them all to end up
target_dir = "/Users/emre/Desktop/DATASETv3/labels/obj_train_data/" 

# Clean the path to remove trailing slashes so os.path.dirname() matches perfectly
target_dir = os.path.normpath(target_dir)

search_pattern = f"{target_dir}/**/*.txt"

print("Flattening dataset... moving txt files to root folder.\n")

moved_count = 0
already_there_count = 0
conflict_count = 0

for file_path in glob.glob(search_pattern, recursive=True):
    filename = os.path.basename(file_path)
    destination = os.path.join(target_dir, filename)
    
    # 1. Skip if the file is already sitting in the main labels folder
    if os.path.dirname(file_path) == target_dir:
        print(f"⏩ Already in root: '{file_path}'")
        already_there_count += 1
        continue
        
    # 2. Prevent the "Duplicate Name" nightmare (Real clashes)
    if os.path.exists(destination):
        print(f"🛑 CLASH DETECTED:")
        print(f"   Source trying to move: {file_path}")
        print(f"   Already exists at:     {destination}")
        conflict_count += 1
        continue
        
    # 3. Move the file
    shutil.move(file_path, destination)
    moved_count += 1

print("\n--- Move Complete ---")
print(f"✅ Successfully Moved: {moved_count}")
print(f"⏩ Skipped (Already in root): {already_there_count}")
print(f"🛑 Skipped (Name conflicts): {conflict_count}")