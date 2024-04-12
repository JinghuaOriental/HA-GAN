'''
Author: JinghuaOriental 1795185859@qq.com
Date: 2024-04-09 11:20:10
LastEditors: JinghuaOriental 1795185859@qq.com
LastEditTime: 2024-04-12 11:06:45
FilePath: /hyj/python_files/Generate/GANs/HAGAN/HA-GAN/preprocess.py
Description: This script is used to preprocess the input images for training and testing the HAGAN model.
'''
# resize and rescale images for preprocessing

import os
import glob
import multiprocessing as mp
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

# print(os.system(". activate hagan"))

### Configs
# 8 cores are used for multi-thread processing
NUM_JOBS = 8
#  resized output size, can be 128 or 256
IMG_SIZE = 256
INPUT_DATA_DIR = '/home/hyj/python_files2/AAA_data/EGFR/CT_112_112_112/0_255_original_niigz/egfr_no_niigz'
OUTPUT_DATA_DIR = '/home/hyj/python_files2/AAA_data/EGFR/CT_112_112_112/egfr_no_preprocessed/'
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
# the intensity range is clipped with the two thresholds, this default is used for our CT images, please adapt to your own dataset
LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600
# suffix (ext.) of input images
SUFFIX = '.nii.gz'
# whether or not to trim blank axial slices, recommend to set as True
TRIM_BLANK_SLICES = True

def resize_img(img):
    """ Resize and rescale the input image """
    nan_mask = np.isnan(img) # Remove NaN
    img[nan_mask] = LOW_THRESHOLD
    img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1,1])

    if TRIM_BLANK_SLICES:
        valid_plane_i = np.mean(img, (1,2)) != -1 # Remove blank axial planes
        img = img[valid_plane_i,:,:]

    img = resize(img, (IMG_SIZE, IMG_SIZE, IMG_SIZE), mode='constant', cval=-1)
    return img

def batch_resize(batch_idx, img_list):
    """ Resize and rescale the input images in a batch """
    # for idx in range(len(img_list)):
    for idx, _ in enumerate(img_list):
        if idx % NUM_JOBS != batch_idx:
            continue
        img_name = img_list[idx].split('/')[-1]
        npy_save_path = os.path.join(OUTPUT_DATA_DIR, f"{img_name.split('.')[0]}.npy")
        if os.path.exists(npy_save_path):
            # skip images that already finished pre-processing
            continue
        try:
            # img = sitk.ReadImage(INPUT_DATA_DIR + img_list[idx])
            img = sitk.ReadImage(img_list[idx])
        except OSError as e: 
            # skip corrupted images
            print(e)
            print("Image loading error:", img_name)
            continue 
        
        img = sitk.GetArrayFromImage(img)
        try:
            img = resize_img(img)
        except OSError as e: # Some images are corrupted
            print(e)
            print("Image resize error:", img_name)
            continue
        # preprocessed images are saved in numpy arrays
        np.save(npy_save_path, img)

def main():
    """ Main function """
    img_list = list(glob.glob(os.path.join(INPUT_DATA_DIR, f"*{SUFFIX}")))

    processes = []
    for i in range(NUM_JOBS):
        processes.append(mp.Process(target=batch_resize, args=(i, img_list)))
    for p in processes:
        p.start()
        p.join()    # 很重要！！

    


if __name__ == '__main__':
    main()
