import segmentation_models_pytorch_1 as smp
from typing import List, Tuple
import h5py
import numpy as np
from argparse import ArgumentParser
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset as BaseDataset
import torch.optim as optim
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from contextlib import suppress
from skimage import exposure
import albumentations as albu
import os
import pickle
from data_aug import data_aug
import json
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from data_aug import MyToTensor, change_to_tensor, change_order, mosaic4
import random
from copy import deepcopy


class SemiDataset(BaseDataset):
    def __init__(
            self, 
            post_path,
            mask_path,
            config,  
            mode = 'train_l',
            stage = 'val',
            model_args=None,
    ):
        with open(post_path, 'rb') as f:
            post = pickle.load(f)  
            
        post = np.transpose(post, (0, 2, 3, 1))
        self.images = post
        if mask_path is not None:
            with open(mask_path, 'rb') as f:
                masks = pickle.load(f) 
        else:
            masks = None
        # print(masks.shape)
        print(post.shape)
        self.masks = masks
        self.post = post
        self.config = config
        self.stage = stage
        self.mode = mode
        self.model_args = model_args
    
    def __getitem__(self, i):
        mask = None
        transform =  transforms.Compose([
            transforms.Resize(512, antialias=True),
            MyToTensor(),
            transforms.Normalize(mean=self.model_args['mean'], std=self.model_args['std'])
        ])
        pre, gap = None, None

        if self.stage == 'val':
            post = self.images[i]
            pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
            pre, image, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))
            image = transform(image)  
            return image, mask
        
        post = self.images[i]
        # print(post.shape)
        pre, image, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
        # mosaic
        # if random.random() < 0.2:
        #     ids = [x for x in range(0, i)] + [x for x in range(i + 1, len(self.images))]
        #     sampled_ids = [i] + random.sample(ids, 3)
        #     imgs = []
        #     for k in range(4):
        #         post = self.images[sampled_ids[k]]
        #         mask = self.masks[sampled_ids[k]]
        #         pre, image, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
        #         imgs.append((pre, image, gap, mask))
        #     pre, image, gap, mask = mosaic4(imgs)

        #     # imsave(f'./imgs/output{i}.png', image[0])
        # else:
        #     post = self.images[i]
        #     pre, image, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
        if self.mode == 'train_l':
            return image, mask
        img_w, img_s1, img_s2 = deepcopy(image), deepcopy(image), deepcopy(image)
        gaussian_transform = transforms.GaussianBlur(5, sigma=(0.1, 2))

        if random.random() < 0.5:
            img_s1 = gaussian_transform(img_s1)
            img_s1 = transform(img_s1)
        cutmix_box1 = obtain_cutmix_box(512, p=0.5)

        if random.random() < 0.5:
            img_s2 = gaussian_transform(img_s2)
            img_s2 = transform(img_s2)
        cutmix_box2 = obtain_cutmix_box(512, p=0.5)

        ignore_mask = torch.zeros((512, 512))
        return img_w, img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

        
    def __len__(self):
        return len(self.images)
    
def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask