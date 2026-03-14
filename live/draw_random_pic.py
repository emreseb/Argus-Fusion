import cv2
import math
import os
import shutil

txt_path = "/home/emre/Desktop/NATO/imgs"
img_path = "/home/emre/Desktop/NATO/coordinates"
    
txt_folder = os.listdir(txt_path)
img_folder = os.listdir(img_path)
    
txt_files = [
    entry for entry in txt_folder
        if os.path.isfile(os.path.join(txt_path,entry))
] 
for filename in txt_files:
    file_path = os.path.join(txt_path,filename)
    
    try:
        with open(txt_path,'r') as file:
            content = file.read()
            pass
    except Exception as e:
        print(f"Man you fucked up cuz like {filename} {e}")





def draw_pic(img):
    pass