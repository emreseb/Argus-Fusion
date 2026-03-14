import numpy as np
from PIL import Image as Image

class Preprocesser:
    def __init__(self, mode):
        self.mode = mode

    def preprocess(self, img_np):
        if self.mode == 'rgbt':
            return rgbt_preprocessing(img_np)
        elif self.mode == 'hsvt':
            return hsvt_preprocessing(img_np)
        else:
            raise ValueError("Unsupported mode. Use 'rgbt' or 'hsvt'.")
    
    def rgbt_preprocessing(img_np):
        img_np[:,:,0] = (img_np[:,:,0] * img_np[:,:,3])/255
        img_np[:,:,1] = (img_np[:,:,1] * img_np[:,:,3])/255
        img_np[:,:,2] = (img_np[:,:,2] * img_np[:,:,3])/255
        return img_np[:,:,:3]


    def hsvt_preprocessing(img_np):
        img_np[:,:,0] = (img_np[:,:,0] * img_np[:,:,3])/255
        img_np[:,:,0] = (img_np[:,:,1] * img_np[:,:,3])/255 #im lazyyyy dawggggg
        img_np[:,:,0] = (img_np[:,:,2] * img_np[:,:,3])/255
        return img_np


    #def main():
    #    img = Image.open("frame_00018.png")
    #    img_np = np.array(img)
    #    is_grayscale = np.all(img_np[:, :, 0] == img_np[:, :, 1]) and np.all(img_np[:, :, 1] == img_np[:, :, 2])
    #    print(is_grayscale)
#
    #    #img_rgb = rgbt_preprocessing(img_np)
    #    #cv2.imshow("RGB Image", img_rgb)
    #    #cv2.waitKey(0)
    #    #cv2.destroyAllWindows()