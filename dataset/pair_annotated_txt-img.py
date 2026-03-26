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
                    
            """We need EO and IR, we need to check if the basename on both of the folder contains the same
            key such as B29 as well as the same frame number such as 00001. If both of them are the same, we can say that they are paired.
            """
            eo_base_name = base_name.__contains__("_0_")
            ir_base_name = base_name.__contains__("_1_")
            
            eo_img_path = os.path.join(img_dir, eo_base_name + '.jpg')       
            ir_img_path = os.path.join(img_dir, ir_base_name + '.jpg')           

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