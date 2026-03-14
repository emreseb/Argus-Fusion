import os

# Configuration
target_dir = '/media/emre/SEB/ROMA'
index_to_change = 2  
remove_index = 5  

def main():
    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} not found.")
        return
    for foldername in os.listdir(target_dir):
        folder_path = os.path.join(target_dir, foldername)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {foldername}")
            rename_files_in_directory(folder_path)
            
def rename_files_in_directory(current_dir):  
    if not os.path.exists(current_dir):
        print(f"Error: Directory {current_dir} not found.")
        return 
              
    for filename in os.listdir(current_dir):
        if filename.endswith(".mp4") and len(filename) > max(index_to_change, remove_index):
            
            new_name = filename[:remove_index] + filename[remove_index+1:]

            if filename[index_to_change] == '2':
                new_name = new_name[:index_to_change] + '1' + new_name[index_to_change+1:]
            
            old_path = os.path.join(current_dir, filename)
            new_path = os.path.join(current_dir, new_name)

            if os.path.exists(new_path) and old_path != new_path:
                print(f"Warning: {new_name} already exists. Skipping.")
                continue

            os.rename(old_path, new_path)

            print(f"Original: {filename}")
            print(f"Target:   {new_name}")
            print("-" * 20)
        else:
            print(f"Skipping: {filename} (Criteria not met)")        
    
if __name__ == "__main__":
    main()