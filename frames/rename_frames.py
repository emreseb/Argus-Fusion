import os

# Configuration
target_dir = '/media/emre/SEB/test'
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
            
def rename_files_in_directory(target_dir):  
    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} not found.")
              
    for filename in os.listdir(target_dir):
        if filename.endswith(".png") and len(filename) > 6:
            
            temp_name = filename
            index_removed = temp_name[:remove_index] + temp_name[remove_index+1:]

            if filename[index_to_change] == '2':
                temp_name = index_removed[:index_to_change] + '1' + index_removed[index_to_change+1:]
            
            old_path = os.path.join(target_dir, filename)
            new_path = os.path.join(target_dir, temp_name)

            os.rename(old_path, new_path)

            print(f"Original: {filename}")
            print(f"Target:   {index_removed}")
            print("-" * 20)
        else:
            print(f"Skipping: {filename} (does not meet criteria)")        
    
if __name__ == "__main__":
    main()