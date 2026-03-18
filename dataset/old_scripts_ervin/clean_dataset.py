import os

# --- CONFIGURATION ---
# UPDATE THIS PATH to point to your 'train' folder seen in Screenshot 1
# Example: r"C:\Users\YourName\Desktop\Drone_Datasets\VTI_Visual_V2\train"
train_path = r"C:\Users\artiarti\Desktop\Drone_Datasets\VTI_Visual_V2\train"

# Paths to the subdirectories
images_dir = os.path.join(train_path, "images")
labels_dir = os.path.join(train_path, "labels")

# List of files to delete (cleaned from your request)
files_to_delete = [
    "DJI_0999_W_JPG.rf.c30acf552ff6de371ed3fe4c23094993.jpg",
    "DJI_0864_W_JPG.rf.2c94b04a80659e9acdc0a68123621dad.jpg",
    "DJI_0684_W_JPG.rf.f927892c8124fabf972f3835cb7a00bd.jpg",
    "DJI_0888_W_JPG.rf.8c9e8665533b998d75ca563b7f2231f9.jpg",
    "DJI_0006_W_JPG.rf.133ce061157fa3a7021a5c9918596229.jpg",
    "DJI_0909_W_JPG.rf.f6b293c8eaa4dab3e0889cc193edb201.jpg",
    "DJI_0906_W_JPG.rf.5175186489fd8eef2a1a52a7048b7b9a.jpg",
    "DJI_0915_W_JPG.rf.3bcd576316d3951dc1343083623c1d4d.jpg",
    "DJI_0867_W_JPG.rf.328e3e012c80e2277164e51feee50f37.jpg",
    "DJI_0930_W_JPG.rf.7e90516c4b9441e84203a139930a21e4.jpg",
    "DJI_0831_W_JPG.rf.61e1721d22a1e670369c6094c6df3295.jpg",
    "DJI_0006_W_JPG.rf.133ce061157fa3a7021a5c9918596229.jpg", # Duplicate in list, script handles it
    "DJI_0717_W_JPG.rf.9a90b12dc742f9f0bf328d5d16225979.jpg",
    "DJI_0870_W_JPG.rf.c1cf0d9644ec8f15bf200c0dc27397b2.jpg",
    "DJI_0531_W_JPG.rf.336e69e345ec2979f61f0978b3f4c833.jpg",
    "DJI_0003_W_JPG.rf.a7b974481bcdfd4b2215a105014bdd75.jpg"
]

# Use a set to remove potential duplicates in the list
unique_files = set(files_to_delete)

print(f"Processing {len(unique_files)} unique files for deletion...\n")

for image_file in unique_files:
    # 1. Construct paths
    image_path = os.path.join(images_dir, image_file)
    
    # Assume label has same name but .txt extension
    # We replace the last occurrence of .jpg just in case
    label_file = image_file.replace(".jpg", ".txt") 
    label_path = os.path.join(labels_dir, label_file)

    # 2. Delete Image
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"[OK] Deleted image: {image_file}")
        except Exception as e:
            print(f"[ERR] Could not delete image {image_file}: {e}")
    else:
        print(f"[MISSING] Image not found: {image_file}")

    # 3. Delete Label
    if os.path.exists(label_path):
        try:
            os.remove(label_path)
            print(f"[OK] Deleted label: {label_file}")
        except Exception as e:
            print(f"[ERR] Could not delete label {label_file}: {e}")
    else:
        print(f"[MISSING] Label not found: {label_file}")

print("\nCleanup complete.")