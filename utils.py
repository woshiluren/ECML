import segmentation_models_pytorch as smp
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

class MyToTensor:
    def __init__(self):
        pass

    def __call__(self, x):
        return x / 255.0


def gpu_state(gpu_id, get_return=False):
    qargs = ['index', 'gpu_name', 'memory.used', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))

    results = os.popen(cmd).readlines()
    gpu_id_list = gpu_id.split(",")
    gpu_space_available = {}
    for cur_state in results:
        cur_state = cur_state.strip().split(", ")
        for i in gpu_id_list:
            if i == cur_state[0]:
                if not get_return:
                    print(f'GPU {i} {cur_state[1]}: Memory-Usage {cur_state[2]} / {cur_state[3]}.')
                else:
                    gpu_space_available[i] = int("".join(list(filter(str.isdigit, cur_state[3])))) - int("".join(list(filter(str.isdigit, cur_state[2]))))
    if get_return:
        return gpu_space_available

def set_gpu(x, space_hold=1000):
    assert torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    torch.backends.cudnn.benchmark = True
    gpu_available = 0
    while gpu_available < space_hold:
        gpu_space_available = gpu_state(x, get_return=True)
        for gpu_id, space in gpu_space_available.items():
            gpu_available += space
        if gpu_available < space_hold:
            gpu_available = 0
            time.sleep(1800) # 间隔30分钟.
    gpu_state(x)

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--configs', type=str, default='configs/Unet_plus_resnet152.yaml')
    parser.add_argument('--configs', type=str, default='configs/DeepLab_resnet152.yaml')
    parser.add_argument('--data_path', type=str, default='dataset/')
    parser.add_argument('--batch_size', type=int , default=8)
    parser.add_argument('--lamda', type=float , default=1)
    parser.add_argument('--semi_batch_size', type=int , default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--max_iteration', type=int, default=2500)
    parser.add_argument('--optimizer', type=str , default='adam')
    parser.add_argument('--loss', type=str , default='DiceLoss')
    parser.add_argument('--gpu', type=str , default='0')
    parser.add_argument('--mode', type=str , default='post')
    parser.add_argument('--seed', type=int , default=1)
    parser.add_argument('--K', type=int , default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float , default=0.0001)
    parser.add_argument('--thres', type=float , default=0.9)
    parser.add_argument('--json_path', type=str , default='aug_config.json')
    args, _ = parser.parse_known_args()
    parser = ArgumentParser(parents=[parser], add_help=False)

    with open(args.configs, "r") as f:
        yaml_params = yaml.safe_load(f)

    with open(args.json_path, "r") as file:
        config = json.load(file)

    for k, v in yaml_params.items():
        setattr(args, k, v)

    return args, parser


def loader(hdf5_file: str, folds: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    post = []
    pre = []
    masks = []
    names = []
    # Read hdf5 file and filter by fold
    with h5py.File(hdf5_file, "r") as f:
        for uuid, values in f.items():
            if 'train' in hdf5_file:
                if values.attrs["fold"] not in folds:
                    continue
            if "pre_fire" not in values:
                continue

            post.append(values["post_fire"][...])
            pre.append(values["pre_fire"][...])
            masks.append(values["mask"][...])
            names.append(uuid)

    # Convert to numpy arrays
    post = np.stack(post, axis=0)
    pre = np.stack(pre, axis=0)
    masks = np.stack(masks, axis=0)
    
    return post, pre, masks, names

class Dataset(BaseDataset):
    def __init__(
            self, 
            pre_path=None,
            post_path=None,
            gap_path=None, 
            mask_path=None,
            config=None, 
            mode = 'post',
            stage = 'val',
            model_args=None,
            california_path = None, 
            K=10
    ):
        pre, post, gap = None, None, None
        if mode in ['pre']:
            with open(pre_path, 'rb') as f:
                pre = pickle.load(f)
            pre = np.transpose(pre, (0, 2, 3, 1))
            self.images = pre
        elif mode in ['post']:
            if stage == 'train':
                post = []
                with open('dataset_pickle/train/fc_post_matrix.pkl', 'rb') as f:
                    post_1 = pickle.load(f)

                    post.append(post_1)
                with open('dataset_pickle/val/fc_post_matrix_folder0.pkl', 'rb') as f:
                    post_2 = pickle.load(f)
                    post.append(post_2)
                post = np.concatenate(post, axis = 0)
            else:
                with open(post_path, 'rb') as f:
                    post = pickle.load(f)
            post = np.transpose(post, (0, 2, 3, 1))
            self.images = post
        elif mode in ['gap']:
            with open(gap_path, 'rb') as f:
                gap = pickle.load(f)
            gap = np.transpose(gap, (0, 2, 3, 1))
            self.images = gap
        elif mode in ['total']:
            with open(pre_path, 'rb') as f:
                pre = pickle.load(f)
            pre = np.transpose(pre, (0, 2, 3, 1))
            with open(post_path, 'rb') as f:
                post = pickle.load(f)
            post = np.transpose(post, (0, 2, 3, 1))

            self.images = np.concatenate((pre, post), axis=-1)
            # raise NotImplementedError("This function is not yet implemented.")
        
        if stage == 'train':
            masks = []
            with open('dataset_pickle/train/train_mask.pkl', 'rb') as f:
                masks_1 = pickle.load(f)
                masks.append(masks_1)
            with open('dataset_pickle/val/val_mask.pkl', 'rb') as f:
                masks_2 = pickle.load(f)
                masks.append(masks_2)
            masks = np.concatenate(masks, axis = 0)
        else:
            with open(mask_path, 'rb') as f:
                masks = pickle.load(f)
            # if stage == 'train':
            #     masks = masks[:278]

        if (california_path is not None) and stage == 'train':
            with open(california_path, 'rb') as f:
                cal_post = pickle.load(f)
            cal_post = np.transpose(cal_post, (0, 2, 3, 1))

            self.images, masks = self.generate_sample(self.images, cal_post, masks, K=K)

        if stage != 'train':
            masks = np.transpose(masks, (0, 2, 3, 1))

        self.masks = masks
        self.post = post
        self.pre = pre
        self.gap = gap
        self.config = config
        self.stage = stage
        self.mode = mode
        self.model_args = model_args
    
    def generate_sample(self, ori_sample, cal_sample, masks, K):
        ori_sample = torch.from_numpy(ori_sample.astype(np.uint8))
        cal_sample = torch.from_numpy(cal_sample.astype(np.uint8))
        masks = torch.from_numpy(masks.astype(np.uint8))

        ori_sample_num = ori_sample.shape[0]
        cal_sample_num = cal_sample.shape[0]

        new_sample_list = []
        new_mask_list = []
        
        for i in range(ori_sample_num):
            print("i:", i)
            new_sample_list.append(ori_sample[i].unsqueeze(0))
            new_mask_list.append(masks[i])

            for j in range(K-1):
                cal_idx = random.randint(0, cal_sample_num-1)

                # 将掩码应用于图像A以获取前景
                foreground = torch.mul(ori_sample[i], masks[i].unsqueeze(0))

                # 将前景复制到图像B上
                result = torch.where(masks[i].unsqueeze(0), foreground, cal_sample[cal_idx])

                new_sample_list.append(result)
                new_mask_list.append(masks[i])
        
        images = torch.cat(new_sample_list, dim=0).numpy()
        masks = torch.stack(new_mask_list, dim=0).numpy()

        return images, masks

    
    def __getitem__(self, i):
        mask = self.masks[i]
        transform =  transforms.Compose([
            transforms.Resize(512, antialias=True),
            MyToTensor(),
            transforms.Normalize(mean=self.model_args['mean'], std=self.model_args['std'])
        ])

        if self.stage == 'train':
            pre, gap, post = None, None, None
            if self.mode in ['pre']:
                # ======================= mosaic 4 ======================== #
                if random.random() < 0.3:
                    ids = [x for x in range(0, i)] + [x for x in range(i + 1, len(self.images))]
                    sampled_ids = [i] + random.sample(ids, 3)
                    imgs = []
                    for k in range(4):
                        pre = self.images[sampled_ids[k]]
                        mask = self.masks[sampled_ids[k]]
                        # pre = preprocess(pre)
                        image, post, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
                        imgs.append((image, post, gap, mask))
                    image, post, gap, mask = mosaic4(imgs)
                else:
                    pre = self.images[i]
                    image, post, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
            elif self.mode in ['post']:
                # ======================= mosaic 4 ======================== #
                if random.random() < 0.3:
                    ids = [x for x in range(0, i)] + [x for x in range(i + 1, len(self.images))]
                    sampled_ids = [i] + random.sample(ids, 3)
                    imgs = []
                    for k in range(4):
                        post = self.images[sampled_ids[k]]
                        mask = self.masks[sampled_ids[k]]
                        post = preprocess(post)
                        pre, image, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
                        imgs.append((pre, image, gap, mask))
                    pre, image, gap, mask = mosaic4(imgs)
                else:
                    post = self.images[i]
                    pre, image, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
            elif self.mode in ['gap']:
                if random.random() < 0.3:
                    ids = [x for x in range(0, i)] + [x for x in range(i + 1, len(self.images))]
                    sampled_ids = [i] + random.sample(ids, 3)
                    imgs = []
                    for k in range(4):
                        gap = self.images[sampled_ids[k]]
                        mask = self.masks[sampled_ids[k]]
                        pre, post, image, mask = data_aug(pre, post, gap, mask, self.model_args)
                        imgs.append((pre, post, image, mask))
                    pre, post, image, mask = mosaic4(imgs)
                else:
                    gap = self.images[i]
                    pre, post, image, mask = data_aug(pre, post, gap, mask, self.model_args)
            elif self.mode in ['total']:
                pre = self.images[i,:,:,:3]
                post = self.images[i,:,:,3:]
                pre, post, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
                image = torch.cat([pre, post], dim=0)
            else:
                raise NotImplementedError("This function is not yet implemented.")
        else:
            pre, post, gap = None, None, None
            if self.mode in ['pre']:
                pre = self.images[i]
                # pre = preprocess(pre)
                pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
                image, post, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))
            elif self.mode in ['post']:
                post = self.images[i]
                # post = preprocess(post)
                pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
                pre, image, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))
            elif self.mode in ['gap']:
                gap = self.images[i]
                # gap = preprocess(gap)
                pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
                pre, post, image, mask = change_order(pre, post, gap, mask, (2, 0, 1))
            elif self.mode in ['total']:
                pre = self.images[i,:,:,:3]
                post = self.images[i,:,:,3:]
                pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
                pre, post, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))
            else:
                raise NotImplementedError("This function is not yet implemented.")

            if self.mode in ['total']:
                pre = transform(pre)
                post = transform(post)
                image = torch.cat([pre, post], dim=0)
            else:
                image = transform(image)    

        return image, mask
    
    def get_length(self):
        return len(self.images)
        
    def __len__(self):
        return len(self.images)

# class Dataset(BaseDataset):
#     def __init__(
#             self, 
#             pre_path,
#             post_path,
#             gap_path, 
#             mask_path,
#             config,  
#             mode = 'post',
#             stage = 'val',
#             model_args=None,
#     ):
#         pre, post, gap = None, None, None
#         if mode in ['pre']:
#             with open(pre_path, 'rb') as f:
#                 pre = pickle.load(f)
#             pre = np.transpose(pre, (0, 2, 3, 1))
#             self.images = pre
#         elif mode in ['post']:
#             if stage == 'train':
#                 with open('dataset_pickle2/addition_feature/train/fc_post_matrix.pkl', 'rb') as f:
#                     post_1 = pickle.load(f)  
#                 with open('dataset_pickle2/addition_feature/train/nbr_post_matrix.pkl', 'rb') as f:
#                     post_2 = pickle.load(f)  
#             else:
#                 with open('dataset_pickle2/addition_feature/val/fc_post_matrix_folder0.pkl', 'rb') as f:
#                     post_1 = pickle.load(f)  
#                 with open('dataset_pickle2/addition_feature/val/nbr_post_matrix_folder0.pkl', 'rb') as f:
#                     post_2 = pickle.load(f) 
#             #nbr
#             post_2 = np.expand_dims(post_2, axis=1)
            
#             post = np.concatenate((post_1, post_2), axis = 1)
#             post = np.transpose(post, (0, 2, 3, 1))
#             self.images = post
#         elif mode in ['gap']:
#             with open(gap_path, 'rb') as f:
#                 gap = pickle.load(f)
#             gap = np.transpose(gap, (0, 2, 3, 1))
#             self.images = gap
#         elif mode in ['total']:
#             raise NotImplementedError("This function is not yet implemented.")
#         with open(mask_path, 'rb') as f:
#             masks = pickle.load(f)[:278]
        
#         self.masks = masks
#         self.post = post
#         self.pre =pre
#         self.gap =gap
#         self.config = config
#         self.stage = stage
#         self.mode = mode
#         self.model_args = model_args
    
#     def __getitem__(self, i):
#         mask = self.masks[i]
#         transform =  transforms.Compose([
#             transforms.Resize(512, antialias=True),
#             MyToTensor(),
#             # transforms.Normalize(mean=self.model_args['mean'].extend([0.5]), std=self.model_args['std'].extend([0.5]))
#             # nbr
#             transforms.Normalize(mean=[0.5 for _ in range(4)], std=[0.5 for _ in range(4)])
#         ])

#         if self.stage == 'train':
#             pre, gap, post = None, None, None
#             if self.mode in ['pre']:
#                 pre = self.images[i]
#                 pre = preprocess(pre)
#                 image, post, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
#             elif self.mode in ['post']:
#                 post = self.images[i]
#                 #post = preprocess(post)
#                 pre, image, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
#             elif self.mode in ['gap']:
#                 gap = self.images[i]
#                 gap = preprocess(gap)
#                 pre, post, image, mask = data_aug(pre, post, gap, mask, self.model_args)
#             else:
#                 raise NotImplementedError("This function is not yet implemented.")
#         else:
#             pre, post, gap = None, None, None
#             if self.mode in ['pre']:
#                 pre = self.images[i]
#                 pre = preprocess(pre)
#                 pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
#                 image, post, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))
#             elif self.mode in ['post']:
#                 post = self.images[i]
#                 #post = preprocess(post)
#                 pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
#                 pre, image, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))
#             elif self.mode in ['gap']:
#                 gap = self.images[i]
#                 # gap = preprocess(gap)
#                 pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
#                 pre, post, image, mask = change_order(pre, post, gap, mask, (2, 0, 1))
#             else:
#                 raise NotImplementedError("This function is not yet implemented.")
#             image = transform(image)    
            
        

#         # print(type(mask))

#         return image, mask
        
#     def __len__(self):
#         return len(self.images)

class SemiDataset(BaseDataset):
    def __init__(
            self, 
            pre_path=None,
            post_path=None,
            gap_path=None, 
            mask_path=None,
            config=None, 
            mode = 'post',
            stage = 'val',
            model_args=None,
            california_path = None, 
            K=10,
            ignore_mask=None
    ):
        pre, post, gap = None, None, None
        if mode in ['pre']:
            with open(pre_path, 'rb') as f:
                pre = pickle.load(f)
            pre = np.transpose(pre, (0, 2, 3, 1))
            self.images = pre
        elif mode in ['post']:
            with open(post_path, 'rb') as f:
                post = pickle.load(f)
            post = np.transpose(post, (0, 2, 3, 1))
            self.images = post
        elif mode in ['gap']:
            with open(gap_path, 'rb') as f:
                gap = pickle.load(f)
            gap = np.transpose(gap, (0, 2, 3, 1))
            self.images = gap
        elif mode in ['total']:
            with open(pre_path, 'rb') as f:
                pre = pickle.load(f)
            pre = np.transpose(pre, (0, 2, 3, 1))
            with open(post_path, 'rb') as f:
                post = pickle.load(f)
            post = np.transpose(post, (0, 2, 3, 1))

            self.images = np.concatenate((pre, post), axis=-1)
            # raise NotImplementedError("This function is not yet implemented.")
        with open(mask_path, 'rb') as f:
            masks = pickle.load(f).cpu()
            # if stage == 'train':
            #     masks = masks[:278]

        if (california_path is not None) and stage == 'train':
            with open(california_path, 'rb') as f:
                cal_post = pickle.load(f)
            cal_post = np.transpose(cal_post, (0, 2, 3, 1))

            self.images, masks = self.generate_sample(self.images, cal_post, masks, K=K)

        self.masks = masks
        self.post = post
        self.pre = pre
        self.gap = gap
        self.config = config
        self.stage = stage
        self.mode = mode
        self.model_args = model_args
        self.ignore_mask = ignore_mask
    
    def generate_sample(self, ori_sample, cal_sample, masks, K):
        ori_sample = torch.from_numpy(ori_sample.astype(np.uint8))
        cal_sample = torch.from_numpy(cal_sample.astype(np.uint8))
        masks = torch.from_numpy(masks.astype(np.uint8))

        ori_sample_num = ori_sample.shape[0]
        cal_sample_num = cal_sample.shape[0]

        new_sample_list = []
        new_mask_list = []
        
        for i in range(ori_sample_num):
            new_sample_list.append(ori_sample[i].unsqueeze(0))
            new_mask_list.append(masks[i])

            for j in range(K-1):
                cal_idx = random.randint(0, cal_sample_num-1)

                # 将掩码应用于图像A以获取前景
                foreground = torch.mul(ori_sample[i], masks[i].unsqueeze(0))

                # 将前景复制到图像B上
                result = torch.where(masks[i].unsqueeze(0), foreground, cal_sample[cal_idx])

                new_sample_list.append(result)
                new_mask_list.append(masks[i])
        
        images = torch.cat(new_sample_list, dim=0).numpy()
        masks = torch.stack(new_mask_list, dim=0).numpy()

        return images, masks

    
    def __getitem__(self, i):
        mask = self.masks[i]
        transform =  transforms.Compose([
            transforms.Resize(512, antialias=True),
            MyToTensor(),
            transforms.Normalize(mean=self.model_args['mean'], std=self.model_args['std'])
        ])

        if self.stage == 'train':
            pre, gap, post = None, None, None
            if self.mode in ['pre']:
                # ======================= mosaic 4 ======================== #
                if random.random() < 0.3:
                    ids = [x for x in range(0, i)] + [x for x in range(i + 1, len(self.images))]
                    sampled_ids = [i] + random.sample(ids, 3)
                    imgs = []
                    for k in range(4):
                        pre = self.images[sampled_ids[k]]
                        mask = self.masks[sampled_ids[k]]
                        # pre = preprocess(pre)
                        image, post, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
                        imgs.append((image, post, gap, mask))
                    image, post, gap, mask = mosaic4(imgs)
                else:
                    pre = self.images[i]
                    image, post, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
            elif self.mode in ['post']:
                # ======================= mosaic 4 ======================== #
                if random.random() < 0.3:
                    ids = [x for x in range(0, i)] + [x for x in range(i + 1, len(self.images))]
                    sampled_ids = [i] + random.sample(ids, 3)
                    imgs = []
                    for k in range(4):
                        post = self.images[sampled_ids[k]]
                        mask = self.masks[sampled_ids[k]]
                        post = preprocess(post)
                        pre, image, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
                        imgs.append((pre, image, gap, mask))
                    pre, image, gap, mask = mosaic4(imgs)
                else:
                    post = self.images[i]
                    pre, image, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
            elif self.mode in ['gap']:
                if random.random() < 0.3:
                    ids = [x for x in range(0, i)] + [x for x in range(i + 1, len(self.images))]
                    sampled_ids = [i] + random.sample(ids, 3)
                    imgs = []
                    for k in range(4):
                        gap = self.images[sampled_ids[k]]
                        mask = self.masks[sampled_ids[k]]
                        pre, post, image, mask = data_aug(pre, post, gap, mask, self.model_args)
                        imgs.append((pre, post, image, mask))
                    pre, post, image, mask = mosaic4(imgs)
                else:
                    gap = self.images[i]
                    pre, post, image, mask = data_aug(pre, post, gap, mask, self.model_args)
            elif self.mode in ['total']:
                pre = self.images[i,:,:,:3]
                post = self.images[i,:,:,3:]
                pre, post, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
                image = torch.cat([pre, post], dim=0)
            else:
                raise NotImplementedError("This function is not yet implemented.")
        else:
            pre, post, gap = None, None, None
            if self.mode in ['pre']:
                pre = self.images[i]
                # pre = preprocess(pre)
                pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
                image, post, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))
            elif self.mode in ['post']:
                post = self.images[i]
                # post = preprocess(post)
                pre, post, gap, _ = change_to_tensor(pre, post, gap, None)
                pre, image, gap, _ = change_order(pre, post, gap, None, (2, 0, 1))
            elif self.mode in ['gap']:
                gap = self.images[i]
                # gap = preprocess(gap)
                pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
                pre, post, image, mask = change_order(pre, post, gap, mask, (2, 0, 1))
            elif self.mode in ['total']:
                pre = self.images[i,:,:,:3]
                post = self.images[i,:,:,3:]
                pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
                pre, post, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))
            else:
                raise NotImplementedError("This function is not yet implemented.")

            if self.mode in ['total']:
                pre = transform(pre)
                post = transform(post)
                image = torch.cat([pre, post], dim=0)
            else:
                image = transform(image)    
        
        ignore_mask = self.ignore_mask[i]

        return image, mask, ignore_mask
    
    def get_length(self):
        return len(self.images)
        
    def __len__(self):
        return len(self.images)


def preprocess(image_matrix):
    image_matrix[image_matrix > 2500] = 2500
    p2, p98 = np.percentile(image_matrix, (2,98))
    image_matrix = exposure.rescale_intensity(image_matrix, in_range=(p2, p98)) / 65535 * 255
    return image_matrix

def prepare_dataloader(args):
    train_post_path = "/data/yic/ecml/dataset_pickle/train_post.pkl"
    train_pre_path = "/data/yic/ecml/dataset_pickle/train_pre.pkl"
    train_mask_path = "/data/yic/ecml/dataset_pickle/train_mask.pkl"
    train_name_path = "/data/yic/ecml/dataset_pickle/train_name.pkl"
    val_post_path = "/data/yic/ecml/dataset_pickle/val_post.pkl"
    val_pre_path = "/data/yic/ecml/dataset_pickle/val_pre.pkl"
    val_mask_path = "/data/yic/ecml/dataset_pickle/val_mask.pkl"
    val_name_path = "/data/yic/ecml/dataset_pickle/val_name.pkl"

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, "imagenet")

    if not (os.path.exists(val_post_path) and os.path.exists(val_pre_path) and os.path.exists(val_mask_path) and os.path.exists(val_name_path)):
        post, pre, masks, names = loader(args.data_path + '/train_eval.hdf5', [0])
        for data_path, data in zip([val_post_path, val_pre_path, val_mask_path, val_name_path], [post, pre, masks, names]):
            f = open(data_path, 'wb')
            pickle.dump(data, f)

    else:
        f1 = open(val_post_path, 'rb')
        f2 = open(val_pre_path, 'rb')
        f3 = open(val_mask_path, 'rb')
        f4 = open(val_name_path, 'rb')
        post = pickle.load(f1)
        pre = pickle.load(f2)
        masks = pickle.load(f3)
        names = pickle.load(f4)
    
    # #数据增强
    # post, pre, masks = data_aug(post, pre, masks, args)
    # print(post.shape)

    val_dataset = Dataset(post[:,:,:,1:4], masks, get_preprocessing(preprocessing_fn))

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    if (os.path.exists(train_post_path) and os.path.exists(train_pre_path) and os.path.exists(train_mask_path) and os.path.exists(train_name_path)):
        # items = os.listdir(args.data_path)
        items = ['train_eval.hdf5', 'california_0.hdf5', 'california_1.hdf5', 'california_2.hdf5', 'california_3.hdf5', 'california_4.hdf5']
        post, pre, masks, names = [], [], [], []
        for item in items:
            if 'hdf5' in item :
                print(item)
                post_, pre_, masks_, names_ = loader(args.data_path + '/' + item, [1,2,3,4,5])
                # if item == 'california_0.hdf5':
                #     print(np.max(masks_[0]))
                #     assert 0
                post.append(post_)
                pre.append(pre_)
                if 'california' in item:
                    masks_ = masks_.transpose(0, 3, 2, 1)
                masks.append(masks_)
                names.append(names_)
        post, pre, masks, names = np.concatenate(post, axis = 0), np.concatenate(pre, axis = 0), np.concatenate(masks, axis = 0), [item for sublist in names for item in sublist]
        # for data_path, data in zip([train_post_path, train_pre_path, train_mask_path, train_name_path], [post, pre, masks, names]):
        #     f = open(data_path, 'wb')
        #     pickle.dump(data, f)
        
    else:
        f1 = open(train_post_path, 'rb')
        f2 = open(train_pre_path, 'rb')
        f3 = open(train_mask_path, 'rb')
        f4 = open(train_name_path, 'rb')
        post = pickle.load(f1)
        pre = pickle.load(f2)
        masks = pickle.load(f3)
        names = pickle.load(f4)

    # 数据增强
    # post, pre, masks = data_aug(post, pre, masks, args)

    train_dataset = Dataset(post[:,:,:,1:4], masks, get_preprocessing(preprocessing_fn))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )
    return train_dataloader, val_dataloader

def prepare_aug_dataloader(args):

    with open(args.json_path, "r") as file:
        config = json.load(file)

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, "imagenet")
    model_params = smp.encoders.get_preprocessing_params(args.encoder_name, 'imagenet')

    # with open('dataset_pickle/fc_post_matrix_folder0.pkl', 'rb') as f:
    #     post = pickle.load(f)
    # with open('dataset_pickle/fc_pre_matrix_folder0.pkl', 'rb') as f:
    #     pre = pickle.load(f)
    # with open('dataset_pickle/val_mask.pkl', 'rb') as f:
    #     masks = pickle.load(f)
    # with open('dataset_pickle/val_name.pkl', 'rb') as f:
    #     names = pickle.load(f)
    # post, pre = np.transpose(post, (0, 2, 3, 1)), np.transpose(pre, (0, 2, 3, 1))
    # post, pre, masks = data_aug(post, pre, masks)

    post_path = 'dataset_pickle/test/fc_post_matrix.pkl'
    # mask_path = 'dataset_pickle/test/test_ensemble_tuned.pkl'
    mask_path = 'dataset_pickle/test/test_mask.pkl'

    # val_dataset = Dataset(post, masks, get_preprocessing(preprocessing_fn))
    val_dataset = Dataset(post_path=post_path, mask_path=mask_path, config=config, mode=args.mode, stage='val', model_args = model_params)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    # items = os.listdir(args.data_path)
    # post, pre, masks, names = [], [], [], []
    # # fc_pkl_path_list = ['dataset_pickle/fc_post_matrix.pkl', 'dataset_pickle/fc_post_matrix_cal.pkl']
    # fc_pkl_path_list = ['dataset_pickle/fc_post_matrix.pkl']
    # # fc_pkl_path_list = ['dataset_pickle/fc_post_matrix_cal.pkl']
    # for fc_pkl_path in fc_pkl_path_list:
    #     with open(fc_pkl_path, 'rb') as f:
    #         post.append(pickle.load(f))
    #     with open(fc_pkl_path.replace("post", "pre"), 'rb') as f:
    #         pre.append(pickle.load(f))
    
    # post = np.concatenate(post, axis=0)
    # pre = np.concatenate(pre, axis=0)

    # with open('dataset_pickle/train_mask.pkl', 'rb') as f:
    #     masks = pickle.load(f)
    # with open('dataset_pickle/train_name.pkl', 'rb') as f:
    #     names = pickle.load(f)
    # post, pre = np.transpose(post, (0, 2, 3, 1)), np.transpose(pre, (0, 2, 3, 1))

    post_path = 'dataset_pickle/train/fc_post_matrix.pkl'
    mask_path = 'dataset_pickle/train/train_mask.pkl'
    california_path = 'dataset_pickle/fc_post_matrix_cal.pkl'

    # train_dataset = Dataset(post_path=post_path, mask_path=mask_path, config=config, mode=args.mode, stage='train', model_args = model_params, california_path=california_path, K=args.K)
    train_dataset = Dataset(post_path=post_path, mask_path=mask_path, config=config, mode=args.mode, stage='train', model_args = model_params)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )
    return train_dataloader, val_dataloader


def prepare_semi_dataloader(args):

    with open(args.json_path, "r") as file:
        config = json.load(file)

    model_params = smp.encoders.get_preprocessing_params(args.encoder_name, 'imagenet')

    post_path = 'dataset_pickle/val/fc_post_matrix_folder0.pkl'
    mask_path = 'final_result/val_ensemble.pkl'

    ens_res_path = "final_result/val_ensemble.pkl"
    f = open(ens_res_path, 'rb')
    ens_res = pickle.load(f)

    # 设置阈值
    threshold = args.thres
    # 根据阈值创建掩码矩阵
    mask_matrix = (ens_res < threshold).cpu()

    # val_dataset = Dataset(post, masks, get_preprocessing(preprocessing_fn))
    semi_dataset = SemiDataset(post_path=post_path, mask_path=mask_path, config=config, mode=args.mode, stage='val', model_args = model_params, ignore_mask=mask_matrix)
    semi_dataloader = DataLoader(
        semi_dataset,
        batch_size=args.semi_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )

    return semi_dataloader