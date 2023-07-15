import numpy as np
import pandas as pd
import h5py
import segmentation_models_pytorch as smp
from trimesh.voxel.runlength import dense_to_brle
from pathlib import Path
from collections import defaultdict
from utils import preprocess
from typing import Any, Union, Dict, Literal
from numpy.typing import NDArray
import torch
from utils import prepare_aug_dataloader, get_command_line_parser
import pickle
import copy
from ensemble import ensemble
import torchvision.transforms as transforms
from data_aug import MyToTensor
def retrieve_validation_fold(path: Union[str, Path]) -> Dict[str, NDArray]:
    result = defaultdict(dict)
    with h5py.File(path, 'r') as fp:
        for uuid, values in fp.items():
            # if values.attrs['fold'] == 0:
            result[uuid]['post'] = values['post_fire'][...]
            
            
            # result[uuid]['pre'] = values['pre_fire'][...]
    return dict(result)

def compute_submission_mask(id: str, mask: NDArray):

    mask = mask.detach().cpu().numpy().flatten().astype(bool)
    # mask = mask.flatten().astype(bool)
    brle = dense_to_brle(mask)
    return {"id": id, "rle_mask": brle, "index": np.arange(len(brle))}

if __name__ == '__main__':

    encoder_name = 'densenet169'
    validation_fold = retrieve_validation_fold('dataset/test.h5')
    result = []
    print(len(validation_fold))
    # model = torch.load('best_model/0608-21-18-26-448/Unet_densenet169_lr0.001_wd0.0001_2_best.pth')
    # model.cuda()
    # model.eval()
    # args, parser = get_command_line_parser()
    # train_dataloader, val_dataloader = prepare_aug_dataloader(args)

    # with torch.no_grad():
    #     for uuid, (input_images, labels) in zip(validation_fold, val_dataloader):
    #         input_images = input_images.cuda()
    #         logits_mask = model(input_images)
    #         prob_mask = logits_mask.sigmoid()
    #         pred_mask = (prob_mask > 0.5).float()
    #         encoded_prediction = compute_submission_mask(uuid, pred_mask)
    #         result.append(pd.DataFrame(encoded_prediction))
    # path_list = {
    #     'unetplusplus_resnet152': '/data/renl/ecml-06-11/best_model/unetplusplus_resnet152_lr0.0001_wd0.0005_lossTverskyLoss_best_1.pth',
    #     'DeepLabV3_resnet152_1': '/data/renl/ecml-06-11/best_model/DeepLabV3_resnet152_lr0.001_wd0.0005_lossDiceLoss_best_1.pth',
    #     'DeepLabV3_resnet152_2': '/data/renl/ecml-06-11/best_model/DeepLabV3_resnet152_lr0.0001_wd0.0005_lossDiceLoss_best_1.pth',
    #     'DeepLabV3Plus_se_resnext101': '/data/renl/ecml-06-11/best_model/DeepLabV3Plus_se_resnext101_32x4d_lr0.001_wd0.0_lossDiceLoss_best_1.pth',
    #     'unetplusplus_densenet169':'/data/renl/ecml-06-11/best_model/unetplusplus_densenet169_lr0.0001_wd0.0_lossDiceLoss_best_1.pth',
    #     'Unet_densenet169': '/data/renl/ecml-06-11/best_model/0614-02-35-22-205/Unet_densenet169_lr0.0001_wd0.0001_2_best.pth',
    # }
    # pkl_list = {
    #     'unetplusplus_resnet152': 'best_result/unetplusplus_resnet152.pkl',
    #     'DeepLabV3_resnet152_1': 'best_result/DeepLabV3_resnet152_1.pkl',
    #     'DeepLabV3_resnet152_2': 'best_result/DeepLabV3_resnet152_2.pkl',
    #     'DeepLabV3Plus_se_resnext101': 'best_result/DeepLabV3Plus_se_resnext101.pkl',
    #     'unetplusplus_densenet169':'best_result/unetplusplus_densenet169.pkl',
    #     'Unet_densenet169': 'best_result/Unet_densenet169.pkl',
    #     'Linknet_densenet169':'best_result/unetplusplus_densenet169.pkl'
    # }


    # order_result = {}
    # ensemble_mode = 1
    # for key, value in pkl_list.items():
    #     with open(value, 'rb') as f:
    #         results = pickle.load(f)
    #     logit = results['logit'].cuda()
    #     target = results['target'].cuda()
    #     logit = logit.sigmoid()
    #     logit = (logit >= 0.5).float()
    #     tp, fp, fn, tn = smp.metrics.get_stats(logit.long(), target.long(), mode="binary")
    #     dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    #     order_result[key] = dataset_iou.cpu()
    # ordered_result = sorted(order_result.items(), key=lambda x: x[1], reverse=True)
    # iou = 0
    # tensor_list = []
    # for key, value in ordered_result:
    #     potential_list = copy.copy(tensor_list)
    #     potential_list.append(pkl_list[key])
    #     potential_logit = []
    #     for path in potential_list:
    #         with open(path, 'rb') as f:
    #             results = pickle.load(f)
    #         logit = results['logit'].cuda()
    #         target = results['target'].cuda()
    #         potential_logit.append(logit)
    #     ensemble_output = ensemble(potential_logit, 0.5, 0.5, ensemble_mode)
    #     tp, fp, fn, tn = smp.metrics.get_stats(ensemble_output.long(), target.long(), mode="binary")
    #     dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    #     if dataset_iou > iou:
    #         iou = dataset_iou
    #         tensor_list = copy.copy(potential_list)
    #         final_result = copy.copy(ensemble_output)

    with open('final_result/test_ensemble_1.pkl', 'rb') as f:
        final_result = pickle.load(f)
    model_args = smp.encoders.get_preprocessing_params('densenet169', 'imagenet')
    transform =  transforms.Compose([
        transforms.Resize(512, antialias=True),
        MyToTensor(),
        transforms.Normalize(mean=model_args['mean'], std=model_args['std'])
        # nbr
        # transforms.Normalize(mean=(0.5), std=(0.5))
    ])
    for uuid, predicted in zip(validation_fold,final_result):
        # convert the prediction in RLE format
        
        encoded_prediction = compute_submission_mask(uuid, predicted)
        result.append(pd.DataFrame(encoded_prediction))

    submission_df = pd.concat(result)
    submission_df.to_csv('result_to_submit/final_predictions.csv', index=False)
