import os

# Update these to your actual paths
paths = [
    "/Users/emre/Desktop/DATASET-(vid)/ROMA/IR",
    "/Users/emre/Desktop/DATASET-(vid)/ROMA/EO"
]

target = "_E"      # Looking for the underscore followed by E
replacement = "_ERF" # Replacing it with underscore followed by ERF

def rename_files_in_folders(folder_paths):
    for folder in folder_paths:
        if not os.path.exists(folder):
            print(f"Directory not found: {folder}")
            continue
            
        print(f"Processing: {folder}")
        count = 0
        
        for filename in os.listdir(folder):
            # Check if the target exists in the filename
            if target in filename:
                new_filename = filename.replace(target, replacement)
                
                old_file_path = os.path.join(folder, filename)
                new_file_path = os.path.join(folder, new_filename)
                
                # Perform the rename
                os.rename(old_file_path, new_file_path)
                count += 1
        
        print(f"Successfully renamed {count} files in {os.path.basename(folder)}.")

if __name__ == "__main__":
    rename_files_in_folders(paths)