import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch_1 as smp
from pprint import pprint
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from utils import loader, prepare_dataloader, get_command_line_parser, set_gpu, prepare_aug_dataloader, prepare_semi_dataloader
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
import torch.nn.functional as F
import warnings
import torch.nn as nn
from tqdm import tqdm

# 忽略所有警告
warnings.filterwarnings("ignore")

class FireModel(pl.LightningModule):

    def __init__(self, args, arch, encoder_name, in_channels, out_classes, ens_res, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        self.args = args
        self.dataset_iou = 0
        self.per_image_iou = 0
        self.ens_res = ens_res

        self.ignore_mask = self.get_ignore_mask()
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        if args.loss == 'JaccardLoss':
            print('JaccardLoss')
            self.loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif args.loss == 'DiceLoss':
            print('DiceLoss')
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif args.loss == 'TverskyLoss':
            print('TverskyLoss')
            self.loss_fn = smp.losses.TverskyLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif args.loss == 'FocalLoss':
            print('FocalLoss')
            self.loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
        elif args.loss == 'LovaszLoss':
            print('LovaszLoss')
            self.loss_fn = smp.losses.LovaszLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif args.loss == 'MCCLoss':
            print('MCCLoss')
            self.loss_fn = smp.losses.MCCLoss()

    def get_ignore_mask(self):
        # 设置阈值
        threshold = 0.5

        # 根据阈值创建掩码矩阵
        mask_matrix = (self.ens_res > threshold).float()

        # 打印掩码矩阵
        print(mask_matrix.shape)
        assert 0

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch[0].type(torch.cuda.HalfTensor)

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        if stage == 'valid':
            import os
            path = Path(f'best_model/{args.time_str}')
            path = Path(f'best_model/{args.time_str}')
            if not path.exists():
                path.mkdir(parents=True)

            if self.dataset_iou < dataset_iou:
                self.dataset_iou = dataset_iou
                torch.save(self.model, f'best_model/{args.time_str}/{args.model_name}_{args.encoder_name}_lr{args.lr}_wd{args.wd}_loss{args.loss}_best_1.pth')

            if self.per_image_iou < per_image_iou:
                self.per_image_iou = per_image_iou
                torch.save(self.model, f'best_model/{args.time_str}/{args.model_name}_{args.encoder_name}_lr{args.lr}_wd{args.wd}_loss{args.loss}_best_2.pth')
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        elif self.args.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.9, dampening=0, weight_decay=self.args.wd, nesterov = True)
        elif self.args.optimizer == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)


def check_data(image, mask):
    image = image.type(torch.cuda.HalfTensor)

    # Shape of the image should be (batch_size, num_channels, height, width)
    # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
    assert image.ndim == 4

    # Check that image dimensions are divisible by 32, 
    # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
    # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
    # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
    # and we will get an error trying to concat these features
    h, w = image.shape[2:]
    assert h % 32 == 0 and w % 32 == 0

    mask = mask

    # Shape of the mask should be [batch_size, num_classes, height, width]
    # for binary segmentation num_classes = 1
    assert mask.ndim == 4

    # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
    assert mask.max() <= 1.0 and mask.min() >= 0


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

    def forward(self, pred_mask, gt_mask, ignore_matrix):
        logits_mask = F.logsigmoid(pred_mask).exp()
        ce_matrix = - gt_mask * torch.log(logits_mask+0.0001) - (1-gt_mask) * torch.log((1-logits_mask+0.0001))
        ignore_matrix = torch.where(ignore_matrix, torch.tensor(0).cuda(), torch.tensor(1).cuda()).float()
        ce_matrix = ce_matrix * ignore_matrix
        ce_loss = torch.mean(ce_matrix, dim=(1,2,3))

        non_zero_num = torch.sum(ignore_matrix, dim=(1,2,3))
        weight_ce_loss = ce_loss * (pow(512, 2) / (non_zero_num + 1))

        # 获取非零元素的索引
        nonzero_indices = torch.nonzero(weight_ce_loss)
        # 统计非零元素个数
        nonzero_count = nonzero_indices.size(0)

        if nonzero_count > 0:
            mean_ce_loss = torch.sum(weight_ce_loss) / nonzero_count
        else:
            mean_ce_loss = torch.sum(weight_ce_loss)

        return mean_ce_loss


def train(args):
    print("args:", args)

    # ==================== get model ==================== #
    in_channels = 3
    out_classes = 1
    model = smp.create_model(
    args.model_name, args.encoder_name, in_channels=in_channels, classes=out_classes)

    model = model.cuda()

    # ==================== get model related param ==================== #
    dataset_iou = 0
    params = smp.encoders.get_preprocessing_params(args.encoder_name)
    # std = torch.tensor(params["std"]).view(1, 3, 1, 1)
    # mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)

    # ==================== get loss ==================== #
    # for image segmentation dice loss could be the best first choice
    if args.loss == 'JaccardLoss':
        print('JaccardLoss')
        loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
    elif args.loss == 'DiceLoss':
        print('DiceLoss')
        loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    elif args.loss == 'TverskyLoss':
        print('TverskyLoss')
        loss_fn = smp.losses.TverskyLoss(smp.losses.BINARY_MODE, from_logits=True)
    elif args.loss == 'FocalLoss':
        print('FocalLoss')
        loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
    elif args.loss == 'LovaszLoss':
        print('LovaszLoss')
        loss_fn = smp.losses.LovaszLoss(smp.losses.BINARY_MODE, from_logits=True)
    elif args.loss == 'MCCLoss':
        print('MCCLoss')
        loss_fn = smp.losses.MCCLoss()
    
    # semi_loss_fn = smp.losses.SoftBCEWithLogitsLoss(reduction='mean', smooth_factor=None, ignore_index=-100, dim=1)
    semi_loss_fn = SoftmaxLoss()

    # ==================== get dataloader ==================== #
    train_dataloader, val_dataloader = prepare_aug_dataloader(args)
    semi_dataloader = prepare_semi_dataloader(args)

    # ==================== get optimizer ==================== #
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.args.lr, momentum=0.9, dampening=0, weight_decay=args.args.wd, nesterov = True)
    elif args.optimizer == 'adamw':
        return torch.optim.AdamW(args.parameters(), lr=args.args.lr, weight_decay=args.args.wd)


    # ==================== start train ==================== #
    best_dataset_iou = 0

    l_dataloader_iter = iter(train_dataloader)
    u_dataloader_iter = iter(semi_dataloader)
    args.train_iteration = int(train_dataloader.dataset.get_length() / args.batch_size) * args.max_epoch
    progress_bar = tqdm(total=args.train_iteration, desc='Training', leave=True)

    l_loss_list = []
    u_loss_list = []
    loss_list = []

    for batch_idx in range(args.train_iteration):
        model.train()
        try:
            inputs_x, targets_x = next(l_dataloader_iter)
        except:
            l_dataloader_iter = iter(train_dataloader)
            inputs_x, targets_x = next(l_dataloader_iter)
        try:
            inputs_u, targets_u, ignore_mask = next(u_dataloader_iter)
        except:
            u_dataloader_iter = iter(semi_dataloader)
            inputs_u, targets_u, ignore_mask = next(u_dataloader_iter)

        inputs_x = inputs_x.cuda()
        targets_x = targets_x.cuda()
        # inputs_u = inputs_u.cuda()
        # targets_u = targets_u.cuda()
        # ignore_mask = ignore_mask.cuda()
        
        l_logits_mask = model(inputs_x)
        # u_logits_mask = model(inputs_u)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        l_loss = loss_fn(l_logits_mask, targets_x)
        # u_loss = semi_loss_fn(u_logits_mask, targets_u, ignore_mask)

        # loss = l_loss + args.lamda * u_loss
        loss = l_loss
        loss_list.append(loss.item())
        l_loss_list.append(l_loss.item())
        # u_loss_list.append(args.lamda * u_loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条显示
        # progress_bar.set_postfix({'Iteration': f'{batch_idx+1}/{args.train_iteration}', 'Loss': loss.item(), 'l_loss': l_loss.item(), 'u_loss': u_loss.item()})
        progress_bar.set_postfix({'Iteration': f'{batch_idx+1}/{args.train_iteration}', 'Loss': loss.item(), 'l_loss': l_loss.item()})
        progress_bar.update(1)

        if batch_idx % 50 == 0:
            print("l_loss:", sum(l_loss_list)/len(l_loss_list))
            # print("u_loss:", sum(u_loss_list)/len(u_loss_list))
            print("loss:", sum(loss_list)/len(loss_list))

            l_loss_list = []
            u_loss_list = []
            loss_list = []

            dataset_iou = val(val_dataloader, model)
        
            if dataset_iou > best_dataset_iou:
                best_dataset_iou = dataset_iou
                import os
                path = Path(f'best_model/{args.time_str}')
                if not path.exists():
                    path.mkdir(parents=True)
                torch.save(model, f'best_model/{args.time_str}/{args.model_name}_{args.encoder_name}_lr{args.lr}_wd{args.wd}_best.pth')

    return best_dataset_iou


def val(val_loader, model):
    # ==================== start validation ==================== #
    model.eval()

    with torch.no_grad():
        results = []
        for imgs, masks in tqdm(val_loader):
            result = {}
            imgs = imgs.cuda()
            masks = masks.cuda()
            logits_mask = model(imgs)

            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            result['tp'], result['fp'], result['fn'], result['tn'] = smp.metrics.get_stats(pred_mask.long(), masks.long(), mode="binary")
            results.append(result)
        
        tp = torch.cat([x["tp"] for x in results])
        fp = torch.cat([x["fp"] for x in results])
        fn = torch.cat([x["fn"] for x in results])
        tn = torch.cat([x["tn"] for x in results])
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        print("dataset_iou:", dataset_iou)

        return dataset_iou


if __name__ == '__main__':
    # ==================== get argument ==================== #
    args, parser = get_command_line_parser()
    args.time_str = datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]

    torch.cuda.set_device(int(args.gpu))
    
    best_dataset_iou = train(args)

    with open('result.csv', 'a') as f:
        f.write(f'{args.time_str},{args.model_name},{args.encoder_name},{args.loss},{args.lr},{args.wd},{args.optimizer},{args.lamda},{args.thres}.{best_dataset_iou} \n')

    # run validation dataset

    # run test dataset
    # test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    # pprint(test_metrics)
