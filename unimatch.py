import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch_1 as smp
from pprint import pprint
from utilss import (
    loader,
    prepare_dataloader,
    get_command_line_parser,
    set_gpu,
    prepare_aug_dataloader,
    prepare_semi_dataloader
)
from datetime import datetime
from pathlib import Path
import ssl
# from UniMatch.model.semseg.deeplabv3plus import DeepLabV3Plus


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


def train(args):
    print("args:", args)

    # ==================== get model ==================== #
    in_channels = 3
    out_classes = 1
    model = smp.create_model(
        args.model_name, args.encoder_name, in_channels=in_channels, classes=out_classes, encoder_weights='ssl'
    )
    torch.cuda.set_device(args.gpu)

    model.cuda()


    # ==================== get model related param ==================== #
    dataset_iou = 0
    per_image_iou = 0

    params = smp.encoders.get_preprocessing_params(args.encoder_name)
    std = torch.tensor(params["std"]).view(1, 3, 1, 1)
    mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)

    # ==================== get loss ==================== #
    # for image segmentation dice loss could be the best first choice
    if args.loss == "JaccardLoss":
        print("JaccardLoss")
        loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
    elif args.loss == "DiceLoss":
        print("DiceLoss")
        loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    elif args.loss == "TverskyLoss":
        print("TverskyLoss")
        loss_fn = smp.losses.TverskyLoss(smp.losses.BINARY_MODE, from_logits=True)
    elif args.loss == "FocalLoss":
        print("FocalLoss")
        loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
    elif args.loss == "LovaszLoss":
        print("LovaszLoss")
        loss_fn = smp.losses.LovaszLoss(smp.losses.BINARY_MODE, from_logits=True)
    elif args.loss == "MCCLoss":
        print("MCCLoss")
        loss_fn = smp.losses.MCCLoss()

    semi_loss_fn = smp.losses.SoftCrossEntropyLoss(
        reduction="mean", smooth_factor=0.1, ignore_index=-100, dim=1
    )

    # ==================== get dataloader ==================== #
    train_dataloader, val_dataloader = prepare_aug_dataloader(args)
    semi_dataloader = prepare_semi_dataloader(args)

    # ==================== get optimizer ==================== #
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.args.lr,
            momentum=0.9,
            dampening=0,
            weight_decay=args.args.wd,
            nesterov=True,
        )
    elif args.args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            args.parameters(), lr=args.args.lr, weight_decay=args.args.wd
        )

    # ==================== start train ==================== #
    l_dataloader_iter = iter(train_dataloader)
    u_dataloader_iter1 = iter(semi_dataloader)
    u_dataloader_iter2 = iter(semi_dataloader)
    batch_num = int(len(train_dataloader.dataset) / args.batch_size)
    args.train_iteration = batch_num * args.max_epoch

    outputs = []
    best_IoU = 0
    for batch_idx in range(args.train_iteration):
        try:
            img_x, mask_x = next(l_dataloader_iter)
        except:
            l_dataloader_iter = iter(train_dataloader)
            img_x, mask_x = next(l_dataloader_iter)
        try:
            (
                img_u_w,
                img_u_s1,
                img_u_s2,
                ignore_mask,
                cutmix_box1,
                cutmix_box2,
            ) = next(u_dataloader_iter1)
            (
                img_u_w_mix,
                img_u_s1_mix,
                img_u_s2_mix,
                ignore_mask_mix,
                _,
                _,
            ) = next(u_dataloader_iter2)
        except:
            u_dataloader_iter1 = iter(semi_dataloader)
            (
                img_u_w,
                img_u_s1,
                img_u_s2,
                ignore_mask,
                cutmix_box1,
                cutmix_box2,
            ) = next(u_dataloader_iter1)
            u_dataloader_iter2 = iter(semi_dataloader)
            (
                img_u_w_mix,
                img_u_s1_mix,
                img_u_s2_mix,
                ignore_mask_mix,
                _,
                _,
            ) = next(u_dataloader_iter2)

        img_x, mask_x = img_x.cuda(), mask_x.cuda()
        img_u_w = img_u_w.cuda()
        img_u_s1, img_u_s2, ignore_mask = (
            img_u_s1.cuda(),
            img_u_s2.cuda(),
            ignore_mask.cuda(),
        )
        cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
        img_u_w_mix = img_u_w_mix.cuda()
        img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
        ignore_mask_mix = ignore_mask_mix.cuda()

        with torch.no_grad():
            model.eval()

            pred_u_w_mix = model(img_u_w_mix).detach()
            conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
            mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

        img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1_mix[
            cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1
        ]
        img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2_mix[
            cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1
        ]

        model.train()

        num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
        # preds, preds_fp = model(img_x), model(img_u_w)
        preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
        # print(preds.shape)
        pred_x, pred_u_w = preds.split([num_lb, num_ulb])
        pred_u_w_fp = preds_fp[num_lb:]

        pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

        pred_u_w = pred_u_w.detach()
        conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
        mask_u_w = pred_u_w.argmax(dim=1)

        mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = (
            mask_u_w.clone(),
            conf_u_w.clone(),
            ignore_mask.clone(),
        )
        mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = (
            mask_u_w.clone(),
            conf_u_w.clone(),
            ignore_mask.clone(),
        )

        mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
        conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
        ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

        mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
        conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
        ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
        # print(pred_x.shape)
        # print(mask_x.shape)
        loss_x = loss_fn(pred_x, mask_x)
        # print(pred_u_s1.shape)
        # print(mask_u_w_cutmixed1.shape)
        mask_u_w_cutmixed1 = mask_u_w_cutmixed1.unsqueeze(1)
        # print(pred_u_s1.shape)
        # print(mask_u_w_cutmixed1.shape)
        loss_u_s1 = semi_loss_fn(pred_u_s1, mask_u_w_cutmixed1)
        loss_u_s1 = loss_u_s1 * (
            (conf_u_w_cutmixed1 >= 0.5) & (ignore_mask_cutmixed1 != 255)
        )
        loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

        loss_u_s2 = semi_loss_fn(pred_u_s2, mask_u_w_cutmixed2)
        loss_u_s2 = loss_u_s2 * (
            (conf_u_w_cutmixed2 >= 0.5) & (ignore_mask_cutmixed2 != 255)
        )
        loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

        loss_u_w_fp = semi_loss_fn(pred_u_w_fp, mask_u_w)
        loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= 0.5) & (ignore_mask != 255))
        loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

        # loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0
        # loss = loss_x
        # loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25) / 1.5
        loss = (loss_x + loss_u_w_fp * 0.2) / 1.2


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prob_mask = pred_x.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask_x.long(), mode="binary"
        )
        outputs.append(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        if (batch_idx + 1) % batch_num == 0:
            epoch = (batch_idx + 1) // batch_num
            # aggregate step metics
            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])

            per_image_iou = smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="micro-imagewise"
            )

            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

            print(
                f"Epoch {epoch}",
                f"Train Per Image IoU: {per_image_iou:.4f}",
                f"Train Dataset IoU: {dataset_iou:.4f}",
            )
            IoU = val(args, val_dataloader, model)
            is_best = IoU > best_IoU
            best_IoU = max(IoU, best_IoU)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': best_IoU,
            }
            # torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                path = Path(f'best_model/{args.time_str}')
                if not path.exists():
                    path.mkdir(parents=True)
                torch.save(checkpoint, f'best_model/{args.time_str}/{args.model_name}_{args.encoder_name}_lr{args.lr}_wd{args.wd}_{2}_best.pth')
    return best_IoU



def val(args, val_loader, model):
    # ==================== start validation ==================== #
    outputs = []
    with torch.no_grad():
        model.eval()
        val_dataloader_iter = iter(val_loader)
        batch_num = int(len(val_loader.dataset) / args.batch_size)
        for batch_idx in range(batch_num):
            imgs, masks = next(val_dataloader_iter)
            imgs = imgs.cuda()
            masks = masks.cuda()
            logits_mask = model(imgs)

            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(), masks.long(), mode="binary"
            )
            outputs.append(
                {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }
            )
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        print(
            f"Val Per Image IoU: {per_image_iou:.4f}",
            f"Val Dataset IoU: {dataset_iou:.4f}",
        )
    return dataset_iou


if __name__ == "__main__":
    # ==================== get argument ==================== #
    args, parser = get_command_line_parser()
    args.time_str = datetime.now().strftime("%m%d-%H-%M-%S-%f")[:-3]

    best_IoU = train(args)


    # run validation dataset

    # run test dataset
    # test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    # pprint(test_metrics)
