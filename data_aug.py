import cv2
import numpy as np
import json
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torch
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import random

class MyToTensor:
    def __init__(self):
        pass

    def __call__(self, x):
        return x / 255.0

class MyGaussianBlurTransform:
    def __init__(self, kernel_size, prob, min_sigma, max_sigma):
        self.kernel_size = [kernel_size, kernel_size]
        self.blur_config_flag = random.random()
        self.prob = prob
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, x):
        if self.blur_config_flag < self.prob:
            gaussian_transform = transforms.GaussianBlur(self.kernel_size, sigma=(self.min_sigma, self.max_sigma))
            return gaussian_transform(x)
        else:
            return x

class MyRotateTransform:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


class MyColorJitterTransform:
    """Rotate by one of the given angles."""

    def __init__(self, brightness_range, saturation_range, contrast_range):
        color_config = (random.randint(10-brightness_range,10+brightness_range)/10, random.randint(10-saturation_range,10+saturation_range)/10, random.randint(10-contrast_range,10+contrast_range)/10)
        self.brightness, self.contrast, self.saturation = color_config

    def __call__(self, x):
        x = F.adjust_brightness(x, self.brightness)
        x = F.adjust_contrast(x, self.contrast)
        x = F.adjust_saturation(x, self.saturation)
        return x


def change_order(pre=None, post=None, gap=None, mask=None, order=(2, 0, 1)):
    if pre is not None:
        pre = pre.permute(*order)
    if post is not None:
        post = post.permute(*order)
    if gap is not None:
        gap = gap.permute(*order)
    if mask is not None:
        mask = mask.permute(*order)
    return pre, post, gap, mask


def change_to_tensor(pre=None, post=None, gap=None, mask=None):
    if pre is not None:
        pre = torch.from_numpy(pre.astype(np.float32))
    if post is not None:
        post = torch.from_numpy(post.astype(np.float32))
    if gap is not None:
        gap = torch.from_numpy(gap.astype(np.float32))
    if mask is not None:
        mask = torch.from_numpy(mask.astype(np.float32))
    return pre, post, gap, mask


def get_input_type_list(pre=None, post=None, gap=None, mask=None):
    input_type_list = []

    if pre is not None:
        input_type_list.append(0)
    if post is not None:
        input_type_list.append(1)
    if gap is not None:
        input_type_list.append(2)
    if mask is not None:
        input_type_list.append(3)

    return input_type_list

def concate_image(pre=None, post=None, gap=None, mask=None, input_type_list=None):
    concate_img_list = []

    for input_type in input_type_list:
        if input_type == 0:
            concate_img_list.append(pre)
        if input_type == 1:
            concate_img_list.append(post)
        if input_type == 2:
            concate_img_list.append(gap)
        if input_type == 3:
            concate_img_list.append(mask)
    
    concate_img = torch.cat(concate_img_list, dim=0)

    return concate_img


def split_image(concate_img, input_type_list):
    channel_num = concate_img.shape[0]
    present_channel_idx = 0

    pre = None
    post = None
    gap = None
    mask = None

    length = concate_img.shape[0] - 1
    for input_type in input_type_list:
        if input_type == 0:
            pre = concate_img[present_channel_idx:present_channel_idx+length]
            present_channel_idx = present_channel_idx + length
        if input_type == 1:
            post = concate_img[present_channel_idx:present_channel_idx+length]
            present_channel_idx = present_channel_idx + length
        if input_type == 2:
            gap = concate_img[present_channel_idx:present_channel_idx+length]
            present_channel_idx = present_channel_idx + length
        if input_type == 3:
            mask = concate_img[present_channel_idx:present_channel_idx+1]
            present_channel_idx = present_channel_idx + 1
    if channel_num == 3:
        post = concate_img

    return pre, post, gap, mask



def data_aug(pre, post, gap, mask, model_args, json_path='aug_config.json'):
    with open(json_path, "r") as file:
        config = json.load(file)

    pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)

    pre, post, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))

    # print(post.shape)
    input_type_list = get_input_type_list(pre, post, gap, mask)

    concate_img = concate_image(pre, post, gap, mask, input_type_list)
    # print(concate_img.shape)
    transform_list = []

    if config["resize"]["enable"]:
        transform_list.append(transforms.Resize(config["resize"]["target_size"], antialias=True))

    if config["random_flip"]["enable"]:
        transform_list.append(transforms.RandomHorizontalFlip(p=config["random_flip"]['prob']))  # 设置水平翻转的概率为0.5
        transform_list.append(transforms.RandomVerticalFlip(p=config["random_flip"]['prob']))

    if config["random_rotation"]["enable"]:
        # transform_list.append(transforms.RandomRotation(degrees=[0, 90, 180, 270]))
        transform_list.append(MyRotateTransform())

    if config["rand_scale_aspect"]["enable"]:
        transform_list.append(transforms.RandomResizedCrop(
            size=(512, 512),
            scale=(config["rand_scale_aspect"]["rich_crop_min_scale"], 1.0),
            ratio=(1.0 - config["rand_scale_aspect"]["rich_crop_aspect_ratio"], 1.0 + config["rand_scale_aspect"]["rich_crop_aspect_ratio"]),
            antialias=True
        ))

    if config["elastic"]["enable"]:
        if random.random() < config["elastic"]["prob"]:
            transform_list.append(transforms.ElasticTransform(alpha=float(config["elastic"]["alpha"]), sigma=float(config["elastic"]["sigma"])))


    transform1 = transforms.Compose(transform_list)
    trans_concate_img = transform1(concate_img)

    pre, post, gap, mask = split_image(trans_concate_img, input_type_list)
    # print(post.shape)
    # =====================================================

    transform_list2 = []

    if config["add_gaussian_noise"]["enable"]:
        transform_list2.append(MyGaussianBlurTransform(config["add_gaussian_noise"]["kernel_size"], config["add_gaussian_noise"]["prob"], config["add_gaussian_noise"]["min_sigma"], config["add_gaussian_noise"]["max_sigma"]))

    if config["random_jitter"]["enable"]:
        brightness_range, saturation_range, contrast_range = config["random_jitter"]["brightness_range"], config["random_jitter"]["saturation_range"], config["random_jitter"]["contrast_range"]
        transform_list2.append(MyColorJitterTransform(config["random_jitter"]["brightness_range"], config["random_jitter"]["saturation_range"], config["random_jitter"]["contrast_range"]))

    transform_list2.append(MyToTensor())
    transform_list2.append(transforms.Normalize(mean=model_args['mean'], std=model_args['std']))
    # transform_list2.append(transforms.Normalize(mean=[0.5 for _ in range(4)], std=[0.5 for _ in range(4)]))

    transform2 = transforms.Compose(transform_list2)

    if not pre is None:
        pre = transform2(pre)
    if not post is None:
        # print(post.shape)
        post = transform2(post)
    if not gap is None:
        post = transform2(gap)
    # if not mask is None:
    #     mask = torch.from_numpy(mask)

    # print("mask.dtype:", mask.dtype)
    # print("mask.shape:", mask.shape)
    # assert 0
    
    return pre, post, gap, mask

def mosaic4(imgs):
    concate_imgs = []
    for img in imgs:
        pre, post, gap, mask = img
        # pre, post, gap, mask = change_to_tensor(pre, post, gap, mask)
        # pre, post, gap, mask = change_order(pre, post, gap, mask, (2, 0, 1))
        input_type_list = get_input_type_list(pre, post, gap, mask)
        concate_img = concate_image(pre, post, gap, mask, input_type_list)
        concate_imgs.append(concate_img)

    # mosaic center
    xc = int(np.random.uniform(128, 384))
    yc = int(np.random.uniform(128, 384))
    mosaic_img = torch.zeros_like(concate_imgs[0])
    for i in range(4):
        if i == 0:  
            #mosaic img: x_min, y_min, x_max, y_max
            x1, y1, x2, y2 = 0, 0, xc, yc
        elif i == 1: 
            x1, y1, x2, y2 = xc + 1, 0, 511, yc
        elif i == 2:
            x1, y1, x2, y2 = 0, yc + 1, xc, 511
        elif i == 3:
            x1, y1 ,x2, y2 = xc + 1, yc + 1, 511, 511
        resize = transforms.Resize((x2 - x1 + 1, y2 - y1 + 1), antialias=True)
        concate_img = concate_imgs[i]
        concate_img = resize(concate_img)
        mosaic_img[:, x1:x2+1, y1:y2+1] = concate_img
    pre, post, gap, mask = split_image(mosaic_img, input_type_list)
    return pre, post, gap, mask
