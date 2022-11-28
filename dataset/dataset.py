import numpy as np
import cv2  # https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        """map()方法返回一个由原数组中的每个元素调用一个指定方法后的返回值组成的新数组
           list() 方法用于将元组转换为列表"""
        self.mask_paths = list(map(lambda x: x.replace('volume', 'segmentation').replace('image', 'mask'), self.img_paths))
        # self.mask_paths = mask_paths
        self.aug = aug

        # print(self.img_paths,self.mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        """numpy.load()函数从具有npy扩展名(.npy)的磁盘文件返回输入数组"""
        npimage = np.load(img_path, allow_pickle=True)
        npmask = np.load(mask_path, allow_pickle=True)
        npimage = npimage[:, :, :, np.newaxis]
        npimage = npimage.transpose((3, 0, 1, 2))

        liver_label = npmask.copy()
        """对于肝脏标记为liver和tumor的标签全部看出是肝脏"""
        liver_label[npmask == 2] = 1
        liver_label[npmask == 1] = 1

        tumor_label = npmask.copy()
        tumor_label[npmask == 1] = 0
        tumor_label[npmask == 2] = 1

        nplabel = np.empty((32, 64, 64, 2))

        nplabel[:, :, :, 0] = liver_label
        nplabel[:, :, :, 1] = tumor_label

        nplabel = nplabel.transpose((3, 0, 1, 2))
        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")
        # print(npimage.shape)

        return npimage, nplabel
