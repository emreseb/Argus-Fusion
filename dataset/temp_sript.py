import os
import glob

# 1. Diagnostic Check
target_dir = "/Users/emre/Desktop/DATASETv3/images"

if not os.path.exists(target_dir):
    print(f"❌ ERROR: Python cannot see this folder at all. Check the path or Mac permissions.")
else:
    print(f"✅ Folder exists! Python can see {len(os.listdir(target_dir))} items inside the top level.")

search_pattern = f"{target_dir}/**/*.jpg"

print("\nScanning for images...")
# 3. Run the loop
count = 0
for file in glob.glob(search_pattern, recursive=True):
    print(file)
    count += 1

print(f"\nFound a total of {count} images.")

