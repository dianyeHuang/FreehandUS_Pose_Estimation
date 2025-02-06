'''
    preprocess the generated images acquired from the simulator
'''

import os
import cv2
import glob
from natsort import natsorted
from color_seg_utils import color_seg
from tqdm import tqdm

import os
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def seg_images():
    folder_path = '/foler_path_to_the_generated_images'
    
    os.chdir(folder_path)
    img_names = glob.glob("*.png")
    img_names = natsorted(img_names)
    
    save_folder = folder_path+'_seg_color'  # color 0 or 1 in each channel
    create_folder_if_not_exists(save_folder)
    
    for img_name in tqdm(img_names):
        imgpath = os.path.join(folder_path, img_name)
        img = cv2.imread(imgpath)
        
        # preprocess
        hue_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        seg_img, _ = color_seg(hue_img, get_centers=False, show=False)
        
        # save_images
        savepath = os.path.join(save_folder, img_name)
        cv2.imwrite(savepath, seg_img)
        
        # cv2.waitKey(33)
        # break
        
    print('done!')


if __name__ == '__main__':
    seg_images()