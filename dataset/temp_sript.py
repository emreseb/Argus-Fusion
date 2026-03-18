import os 
from pathlib import Path

root_dir = Path("/path/to/your/dataset") # Update this to your dataset directory

for file_path in root_dir.rglob('*.jpg'):
    if file_path.is_file(): # Ensure it's a file, not a directory
        print(file_path)