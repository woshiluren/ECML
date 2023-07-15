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
from semi import SemiDataset


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
    parser.add_argument('--configs', type=str, default='configs/DeepLabV3Plus_se_resnext101.yaml')
    parser.add_argument('--data_path', type=str, default='dataset/')
    parser.add_argument('--batch_size', type=int , default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--optimizer', type=str , default='adam')
    parser.add_argument('--gpu', type=int , default=0)
    parser.add_argument('--seed', type=int , default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float , default=0.0001)
    parser.add_argument('--json_path', type=str , default='aug_config.json')
    parser.add_argument('--activation', type=str , default='sigmoid')
    parser.add_argument('--loss', type=str , default='DiceLoss')
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
    post = np.stack(post, axis=0, dtype=np.int32)
    pre = np.stack(pre, axis=0, dtype=np.int32)
    masks = np.stack(masks, axis=0, dtype=np.int32)
    
    return post, pre, masks, names

class Dataset(BaseDataset):
    def __init__(
            self, 
            pre_path,
            post_path,
            gap_path, 
            mask_path,
            config,  
            mode = 'post',
            stage = 'val',
            model_args=None,
    ):
        pre, post, gap = None, None, None
        if mode in ['pre']:
            with open(pre_path, 'rb') as f:
                pre = pickle.load(f)
            pre = np.transpose(pre, (0, 2, 3, 1))
            self.images = pre
        elif mode in ['post']:
            # if stage == 'train':
            #     post = []
            #     with open('dataset_pickle_2/train/fc_post_matrix_folder0.pkl', 'rb') as f:
            #         post_1 = pickle.load(f)
            #         post.append(post_1)
            #     with open('dataset_pickle_2/val/fc_post_matrix_folder0.pkl', 'rb') as f:
            #         post_2 = pickle.load(f)
            #         post.append(post_2)
            #     post = np.concatenate(post, axis = 0)
            # else:                
            with open(post_path, 'rb') as f:
                post = pickle.load(f)  
            
            #nbr
            # post = np.expand_dims(post, axis=1)
            post = np.transpose(post, (0, 2, 3, 1))
            self.images = post
        elif mode in ['gap']:
            with open(gap_path, 'rb') as f:
                gap = pickle.load(f)
            gap = np.transpose(gap, (0, 2, 3, 1))
            self.images = gap
        elif mode in ['total']:
            raise NotImplementedError("This function is not yet implemented.")
        # if stage == 'train':
        #     masks = []
        #     with open('dataset_pickle_2/train/train_mask.pkl', 'rb') as f:
        #         masks_1 = pickle.load(f)
        #         masks.append(masks_1)
        #     with open('dataset_pickle_2/val/val_mask.pkl', 'rb') as f:
        #         masks_2 = pickle.load(f)
        #         masks.append(masks_2)
        #     masks = np.concatenate(masks, axis = 0)
        # else:
        with open(mask_path, 'rb') as f:
            masks = pickle.load(f) 
        print(masks.shape)
        print(post.shape)
        self.masks = masks
        self.post = post
        self.pre =pre
        self.gap =gap
        self.config = config
        self.stage = stage
        self.mode = mode
        self.model_args = model_args
    
    def __getitem__(self, i):
        mask = self.masks[i]
        transform =  transforms.Compose([
            transforms.Resize(512, antialias=True),
            MyToTensor(),
            transforms.Normalize(mean=self.model_args['mean'], std=self.model_args['std'])
            # nbr
            # transforms.Normalize(mean=(0.5), std=(0.5))
        ])

        if self.stage == 'train':
            pre, gap, post = None, None, None
            if self.mode in ['pre']:
                if random.random() < 0.2:
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
                if random.random() < 0.2:
                    ids = [x for x in range(0, i)] + [x for x in range(i + 1, len(self.images))]
                    sampled_ids = [i] + random.sample(ids, 3)
                    imgs = []
                    for k in range(4):
                        post = self.images[sampled_ids[k]]
                        mask = self.masks[sampled_ids[k]]
                        # post = preprocess(post)
                        pre, image, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
                        imgs.append((pre, image, gap, mask))
                    pre, image, gap, mask = mosaic4(imgs)

                    # imsave(f'./imgs/output{i}.png', image[0])
                else:
                    post = self.images[i]
                    #post = preprocess(post)
                    pre, image, gap, mask = data_aug(pre, post, gap, mask, self.model_args)
            elif self.mode in ['gap']:
                if random.random() < 0.2:
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
                    # gap = preprocess(gap)
                    pre, post, image, mask = data_aug(pre, post, gap, mask, self.model_args)
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
                #post = preprocess(post)
                pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
                pre, image, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))
            elif self.mode in ['gap']:
                gap = self.images[i]
                # gap = preprocess(gap)
                pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
                pre, post, image, mask = change_order(pre, post, gap, mask, (2, 0, 1))
            else:
                raise NotImplementedError("This function is not yet implemented.")
            image = transform(image)    

        return image, mask
        
    def __len__(self):
        return len(self.images)


def preprocess(image_matrix):
    image_matrix[image_matrix > 2500] = 2500
    p2, p98 = np.percentile(image_matrix, (2,98))
    image_matrix = exposure.rescale_intensity(image_matrix, in_range=(p2, p98)) / 65535 * 255
    return image_matrix

def prepare_dataloader(args):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, "ssl")
    with open('dataset_pickle/val_post.pkl', 'rb') as f:
        post = pickle.load(f)
    with open('dataset_pickle/val_pre.pkl', 'rb') as f:
        pre = pickle.load(f)
    # with open('dataset_pickle/nbr_post_matrix_folder0.pkl', 'rb') as f:
    #     post_1 = pickle.load(f)
    # with open('dataset_pickle/nbr_gap_matrix_folder0.pkl', 'rb') as f:
    #     post_2 = pickle.load(f)
    # with open('dataset_pickle/nbr_pre_matrix_folder0.pkl', 'rb') as f:
    #     post_3 = pickle.load(f)
    with open('dataset_pickle/val_mask.pkl', 'rb') as f:
        masks = pickle.load(f)
    with open('dataset_pickle/val_name.pkl', 'rb') as f:
        names = pickle.load(f)

    

    val_dataset = Dataset(post[:,:,:,1:4], masks, get_preprocessing(preprocessing_fn))

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    items = os.listdir(args.data_path)
    post, pre, masks, names = [], [], [], []

    with open('dataset_pickle/train_post.pkl', 'rb') as f:
        post = pickle.load(f)
    with open('dataset_pickle/train_pre.pkl', 'rb') as f:
        pre = pickle.load(f)
    with open('dataset_pickle/train_mask.pkl', 'rb') as f:
        masks = pickle.load(f)
    with open('dataset_pickle/train_name.pkl', 'rb') as f:
        names = pickle.load(f)
    

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
    model_params = smp.encoders.get_preprocessing_params(args.encoder_name, 'ssl')


    val_dataset = Dataset(None, 'dataset_pickle/test/fc_post_matrix.pkl', None, 'dataset_pickle/test/test_mask_.pkl', config, mode = 'post', stage = 'val', model_args = model_params)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        
    )

    train_dataset = Dataset(None, 'dataset_pickle/train/train_val_fc_post_matrix.pkl', None,'dataset_pickle/train/train_val_mask.pkl', config, mode = 'post', stage = 'train', model_args = model_params)
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
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, "imagenet")
    model_params = smp.encoders.get_preprocessing_params(args.encoder_name, 'ssl')

    train_dataset = SemiDataset('dataset_pickle/test/fc_post_matrix.pkl', None, config, mode = 'train_u', stage = 'train', model_args = model_params)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )
    return train_dataloader