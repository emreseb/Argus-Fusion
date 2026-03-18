import os

# ================= CONFIGURATION =================
DATASET_ROOT = r"C:\Data\Reduced_Clean_Dataset_V6"
DELETE_LIST_FILE = "files_to_delete_V6.txt"
# =================================================

def main():
    if not os.path.exists(DELETE_LIST_FILE):
        print(f"No {DELETE_LIST_FILE} found. Nothing to delete.")
        return

    with open(DELETE_LIST_FILE, 'r') as f:
        files_to_delete = [line.strip() for line in f.readlines() if line.strip()]

    if not files_to_delete:
        print("Delete list is empty.")
        return

    print(f"Found {len(files_to_delete)} files marked for deletion.")
    confirm = input("Are you sure you want to PERMANENTLY delete them? (yes/no): ")
    
    if confirm.lower() != "yes":
        print("Aborted.")
        return

    # Subsets to search in
    subsets = ['train', 'valid', 'test']
    
    deleted_count = 0
    
    for filename in files_to_delete:
        found = False
        
        # We don't know if the file is in train, valid, or test, so we check all
        for subset in subsets:
            img_path = os.path.join(DATASET_ROOT, subset, "images", filename)
            
            # Check txt and jpg/png extensions for labels
            label_name = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(DATASET_ROOT, subset, "labels", label_name)

            if os.path.exists(img_path):
                found = True
                try:
                    # Delete Image
                    os.remove(img_path)
                    
                    # Delete Label (if exists)
                    if os.path.exists(label_path):
                        os.remove(label_path)
                        
                    print(f"[DELETED] {subset}: {filename}")
                    deleted_count += 1
                    break # Stop checking other folders once found
                except Exception as e:
                    print(f"[ERROR] Could not delete {filename}: {e}")
        
        if not found:
            print(f"[NOT FOUND] Could not locate {filename} in any folder.")

    print(f"\nCleanup Complete. Deleted {deleted_count} pairs.")
    
    # Optional: Clear the list file after success
    # os.remove(DELETE_LIST_FILE)

if __name__ == "__main__":
    main()