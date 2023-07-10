import os
import numpy as np
import h5py
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import load_split

class NyuDepthLoader(data.Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0) # HWC
        #print('img',img.shape)
        dpt = self.dpts[img_idx].transpose(1, 0)
        img = Image.fromarray(img)
        dpt = Image.fromarray(dpt)

        input_transform = transforms.Compose([transforms.Resize(228),
                                              transforms.ToTensor()])

        target_depth_transform = transforms.Compose([transforms.Resize([128,160]),
                                                     transforms.ToTensor()])

        img = input_transform(img)
        #print(img.size())
        dpt = target_depth_transform(dpt)
        #print(dpt.size())
        return img, dpt

    def __len__(self):
        return len(self.lists)
