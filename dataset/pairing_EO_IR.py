import os
import shutil

def pair_annotated_images(txt_dir, img_dir, output_dir):
    """ Pair annotated EO and IR images based on their filenames. """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    paired_count = 0
    missing_count = 0

    # 1. Grab all text files once so we aren't constantly reading the hard drive
    all_txt_files = set(os.listdir(txt_dir))

    for filename in all_txt_files: 
        # 2. Use EO files as the Anchor. Skip IR files so we don't double-count.
        if filename.endswith('.txt') and "_0_" in filename:
            
            eo_base_name = os.path.splitext(filename)[0]
            
            # 3. Calculate the exact name of the IR twin
            ir_base_name = eo_base_name.replace("_0_", "_1_")
            ir_filename = ir_base_name + ".txt"
            
            # 4. Check if the IR text file actually exists in the folder
            if ir_filename in all_txt_files:
                
                # 5. Build the image paths
                eo_img_path = os.path.join(img_dir, eo_base_name + '.jpg')       
                ir_img_path = os.path.join(img_dir, ir_base_name + '.jpg')
                
                # 6. Final safety check: Do both images actually exist?
                if os.path.exists(eo_img_path) and os.path.exists(ir_img_path):
                    paired_count += 1
                    shutil.copy(eo_img_path, output_dir)
                    shutil.copy(ir_img_path, output_dir)
                    shutil.copy(os.path.join(txt_dir, filename), output_dir)
                    shutil.copy(os.path.join(txt_dir, ir_filename), output_dir)
                    
                else:
                    print(f"⚠️ Images missing for pair: {eo_base_name}")
                    missing_count += 1
            else:
                print(f"⚠️ Missing IR text label for: {eo_base_name}")
                missing_count += 1

    print("\n--- Pairing Complete ---")
    print(f"✅ Successfully Paired: {paired_count} sets")
    print(f"🛑 Missing/Incomplete: {missing_count} sets")

 
def main():
    txt_directory = "/home/emre/Desktop/NATO/DATASETv3/labels/obj_train_data"
    img_directory = "/home/emre/Desktop/NATO/DATASETv3/images"
    output_directory = "/home/emre/Desktop/NATO/DATASETv3/paired_output"
    
    if not os.path.exists(txt_directory):
        print(f"Error: Text directory '{txt_directory}' does not exist.")
        return
    if not os.path.exists(img_directory):
        print(f"Error: Image directory '{img_directory}' does not exist.")
        return
    
    pair_annotated_images(txt_directory, img_directory, output_directory) 
    
if __name__ == "__main__":    main()