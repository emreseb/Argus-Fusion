import os
import shutil

clean_empty = False

def clean_empty_txt_files(txt_dir):
    """ Remove empty txt files in the given directory. """
    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(txt_dir, filename)
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                print(f"Removed empty file: {file_path}")
                
def pair_annotated_images(txt_dir, img_dir, output_dir):
    """ Pair annotated EO and IR images based on their filenames. """
    done =0
    undone = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(txt_dir): 
        if filename.endswith('.txt'):
            base_name = os.path.splitext(filename)[0]
            
            # Checks for the images based on the text file's base name
            eo_img_path = os.path.join(img_dir, f"{base_name}.jpg")
            ir_img_path = os.path.join(img_dir, f"{base_name}.jpg")
            txt_file_path = os.path.join(txt_dir, filename)
            
            if os.path.exists(eo_img_path) and os.path.exists(ir_img_path):

                pair_folder = os.path.join(output_dir, base_name)
                os.makedirs(pair_folder, exist_ok=True)
                
                shutil.copy(eo_img_path, pair_folder)
                shutil.copy(ir_img_path, pair_folder)
                shutil.copy(txt_file_path, pair_folder)
                
                done += 1
                print(f"📁 Created folder and paired: {base_name}")
                
            else:
                undone += 1
                print(f"Missing pair for: {filename}")
                
    print(f"Total {done} files paired successfully.")
    print(f"Total {undone} files without pairs.")
                

def main():
    txt_dir = '/home/emre/Desktop/NATO/DATASETv3/labels/obj_train_data'
    img_dir = '/home/emre/Desktop/NATO/DATASETv3/images'
    output_dir = '/home/emre/Desktop/NATO/DATASETv3/paired_output'

    if clean_empty:
        print("Cleaning empty text files...")
        clean_empty_txt_files(txt_dir)
    
    print("Starting pairing process...")
    pair_annotated_images(txt_dir, img_dir, output_dir)
    print("Process complete!")

if __name__ == "__main__":
    main()