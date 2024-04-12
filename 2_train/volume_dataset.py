'''
Author: JinghuaOriental 1795185859@qq.com
Date: 2024-04-12 15:02:19
LastEditors: JinghuaOriental 1795185859@qq.com
LastEditTime: 2024-04-12 17:21:16
FilePath: /python_files2/Generate/GANs/HAGAN/HA-GAN/2_train/volume_dataset.py
Description: This file is used to load the dataset for training and validation.
'''
import os
import glob
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


import numpy as np
from scipy.ndimage import zoom
from torchvision import transforms

class ResizeArray(object):
    """ Resize an array to a target shape. """
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, array):
        # 使用zoom函数调整大小，将目标形状作为参数传递
        resized_array = zoom(array, self.target_shape / np.array(array.shape))
        return resized_array

class Volume_Dataset(Dataset):
    """ Volume dataset for training and validation. """
    def __init__(self, data_dir, mode='train', fold=0, num_class=0, transform=None):
        self.sid_list = []
        self.data_dir = data_dir
        self.num_class = num_class
        self.transform = transform

        for item in glob.glob(os.path.join(self.data_dir, "*.npy")):
            self.sid_list.append(item.split('/')[-1])

        self.sid_list.sort()
        self.sid_list = np.asarray(self.sid_list)

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        train_index, valid_index = list(kf.split(self.sid_list))[fold]
        print("Fold:", fold)
        if mode=="train":
            self.sid_list = self.sid_list[train_index]
        else:
            self.sid_list = self.sid_list[valid_index]
        print("Dataset size:", len(self))

        self.class_label_dict = dict()
        if self.num_class > 0: # conditional
            with open("class_label.csv", "r") as f:
                f.readline() # header
                for myline in f.readlines():
                    mylist = myline.strip("\n").split(",")
                    self.class_label_dict[mylist[0]] = int(mylist[1])

    def __len__(self):
        return len(self.sid_list)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.data_dir, self.sid_list[idx]))
        if self.transform is not None:
            img = self.transform(img)
        class_label = self.class_label_dict.get(self.sid_list[idx], -1) # -1 if no class label
        return img[None,:,:,:], class_label
