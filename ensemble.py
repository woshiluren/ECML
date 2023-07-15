import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import segmentation_models_pytorch as smp
from pprint import pprint
from torch.utils.data import DataLoader
from utils import loader, prepare_aug_dataloader, get_command_line_parser, set_gpu
import pickle
import copy
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from data_aug import MyToTensor, change_to_tensor, change_order, mosaic4
def ensemble(input, bi_threshold, avg_threshold, ensemble_mode = 1): # input: tensor list, mode: 0 vote, 1 avg
    stacked_tensor = torch.stack(input)
    print(stacked_tensor.shape)
    if ensemble_mode == 0:
        stacked_tensor = stacked_tensor.sigmoid()
        binary_tensor = (stacked_tensor >= bi_threshold).float()
        ensemble_output = torch.mode(binary_tensor, dim=0).values
        return ensemble_output
    else:
        ensemble_output = torch.mean(stacked_tensor, dim=0, keepdim=True)
        ensemble_output = torch.squeeze(ensemble_output, dim=0)
        ensemble_output = ensemble_output.sigmoid()
        binary_output = (ensemble_output >= avg_threshold).float()
        return binary_output



if __name__ == '__main__':
    args, parser = get_command_line_parser()
    train_dataloader, val_dataloader = prepare_aug_dataloader(args)

    # weight = {
    #     'model_1': 0.7391275763511658,
    #     'model_2': 0.7409109473228455,
    #     'model_3': 0.7409109473228455,
    #     'model_4': 0.73810875415802,
    #     'model_5': 0.7344942092895508,
    #     'model_6': 0.7409409880638123,
    #     'model_7': 0.7311689853668213,
    # }


    path_list = {
        # 'best_test_model_1': 'final_model_2/DeepLabV3Plus_resnet152_lr0.0001_wd0.0001_best.pth',
        # 'best_test_model_2': 'final_model_2/DeepLabV3Plus_se_resnext101_32x4d_lr0.0001_wd0.0001_best.pth',
        # 'best_test_model_3': 'final_model_2/unetplusplus_densenet169_lr0.0001_wd0.0001_best.pth',
        # 'best_test_model_4': 'final_model_2/unetplusplus_densenet201_lr0.0001_wd0.0001_best.pth',
        # 'best_test_model_5': 'final_model_2/unetplusplus_se_resnext101_32x4d_lr0.0001_wd0.0001_best.pth',
        # 'best_test_model_6': 'final_model_2/unetplusplus_resnet152_lr0.0001_wd0.0001_best.pth'
        # 'best_test_model_7': 'final_model_2/DeepLabV3Plus_resnext101_32x4d_lr0.0001_wd0.0001_iou0.7771_best.pth'


    #     # 'best_val_model_1': 'val_best_model_ensemble/DeepLabV3Plus_resnet152_lr0.0001_wd0.0001_best_0.7671.pth',
    #     # 'best_val_model_2': 'val_best_model_ensemble/DeepLabV3Plus_se_resnext101_32x4d_lr0.0001_wd0.0001_best_0.7623.pth',
    #     # 'best_val_model_3': 'val_best_model_ensemble/unetplusplus_densenet169_lr0.0001_wd0.0001_best_0.7669.pth',
    #     # 'best_val_model_4': 'val_best_model_ensemble/unetplusplus_se_resnext101_32x4d_lr0.0001_wd0.0001_best_0.7715.pth',
    #     'best_val_model_5': 'val_best_model_ensemble/DeepLabV3Plus_resnext101_32x4d_lr0.0001_wd0.0001_2_best.pth',
    #     # 'best_val_model_6': 'val_best_model_ensemble/unetplusplus_densenet201_lr0.0001_wd0.0001_best.pth'

    #     # 'best_test_model_1': 'final_model_1/DeepLabV3Plus_resnet152_lr0.0001_wd0.0001_best.pth',
    #     # 'best_test_model_2': 'final_model_1/DeepLabV3Plus_se_resnext101_32x4d_lr0.0001_wd0.0001_best.pth',
    #     # 'best_test_model_3': 'final_model_1/unetplusplus_densenet169_lr0.0001_wd0.0001_best.pth',
    #     # 'best_test_model_4': 'final_model_1/unetplusplus_resnet152_lr0.0001_wd0.0001_best.pth',
    #     # 'best_test_model_5': 'final_model_1/unetplusplus_se_resnext101_32x4d_lr0.0001_wd0.0001_best.pth',
    #     # 'best_test_model_6': 'final_model_1/unetplusplus_densenet201_lr0.0001_wd0.0001_best.pth'
    #     # # 'model_11': 'val_best_model/DeepLabV3Plus_resnext101_32x4d_lr0.0001_wd0.0001_2_best.pth',
    #     # 'model_12': 'val_best_model/DeepLabV3Plus_se_resnext101_32x4d_lr0.001_wd0.0001_best.pth',
    #     # 'model_13': 'val_best_model/DeepLabV3Plus_resnet152_lr0.0001_wd0.0001_best.pth',
    #     # 'model_8': 'best_model/0618-15-15-43-246/PSPNet_densenet169_lr0.001_wd0.0001_best.pth',
    # #     # 'model_1': 'val_best_model/0618-01-40-13-862/DeepLabV3Plus_se_resnext101_32x4d_lr0.001_wd0.001_best.pth',
    # #     # 'model_2': 'val_best_model/0618-07-57-06-832/DeepLabV3Plus_se_resnext101_32x4d_lr0.001_wd0.0001_best.pth',
    # #     # 'model_3': 'val_best_model/0618-09-51-45-417/PSPNet_densenet169_lr0.001_wd0.0_best.pth',
    # #     # 'model_4': 'val_best_model/FPN_densenet169_lr0.001_wd0.001_best.pth',
    # #     # 'model_5':'val_best_model/Unet_densenet169_lr0.001_wd0.001_best.pth',
    # #     # 'model_6': 'best_model/0618-11-59-19-142/unetplusplus_densenet169_lr0.001_wd0.0_best.pth',
    # #     # 'model_7': 'best_model/0618-12-07-42-901/DeepLabV3Plus_se_resnext101_32x4d_lr0.001_wd0.0_best.pth',
        
    # #     'model_9': 'best_model/0618-14-16-07-217/DeepLabV3_resnet152_lr0.0001_wd0.001_best.pth',
    # #     'model_10': 'best_model/0618-15-26-15-091/DeepLabV3Plus_se_resnext101_32x4d_lr0.001_wd0.0001_best.pth',
    #     # 'best_val_model_1': 'val_best_model_ensemble/DeepLabV3Plus_resnet152_lr0.0001_wd0.0001_best_0.7671.pth',
    #     # 'best_val_model_2': 'val_best_model_ensemble/DeepLabV3Plus_se_resnext101_32x4d_lr0.0001_wd0.0001_best_0.7623.pth',
    #     # 'best_val_model_3': 'val_best_model_ensemble/unetplusplus_densenet169_lr0.0001_wd0.0001_best_0.7669.pth',
    #     # 'best_val_model_4': 'val_best_model_ensemble/unetplusplus_se_resnext101_32x4d_lr0.0001_wd0.0001_best_0.7715.pth',
    #     # 'best_val_model_5': 'val_best_model_ensemble/DeepLabV3Plus_resnext101_32x4d_lr0.0001_wd0.0001_2_best.pth'
    }

    #test
    # with open('dataset_pickle_2/test/fc_post_matrix_folder0.pkl', 'rb') as f:
    #     post = pickle.load(f)
    # model_args = smp.encoders.get_preprocessing_params('resnet152', 'imagenet')
    # for key, path in path_list.items():
    #     model  = smp.create_model('DeepLabV3Plus', encoder_name='resnext101_32x4d', encoder_weights = None,  in_channels=3, classes=1)
    #     checkpoint = torch.load(path, map_location = 'cpu')['model']
    #     model.load_state_dict(checkpoint)
    #     # # print(model)
    #     # model = torch.load(path, map_location = 'cpu')
    #     model.cuda()
    #     model.eval()
    #     logit = []
    #     with torch.no_grad():
    #         for idx, input in enumerate(post):
    #             _, image, _, _ = change_to_tensor(None, input, None, None)
    #             # _, image, _, _ = change_order(None, input, None, None, (2, 0, 1))

    #             transform =  transforms.Compose([
    #                 transforms.Resize(512, antialias=True),
    #                 MyToTensor(),
    #                 transforms.Normalize(mean=model_args['mean'], std=model_args['std'])
    #                 # nbr
    #                 # transforms.Normalize(mean=(0.5), std=(0.5))
    #             ])
    #             print(image.shape)
    #             image = transform(image)
    #             image = image.cuda()
    #             image = torch.unsqueeze(image, dim = 0)
    #             logits_mask = model(image)
    #             logit.append(logits_mask)
    #         logit_ = torch.cat(logit, dim = 0)
    #     result = {
    #         'logit': logit_,
    #     }
    #     with open(f'final_result_2/{key}.pkl', 'wb') as file:
    #         pickle.dump(result, file)
    #     print(f'{key} saved')



    # val
    # for key, path in path_list.items():
    #     model  = smp.create_model('DeepLabV3Plus', encoder_name='resnext101_32x4d', encoder_weights = None,  in_channels=3, classes=1)
    #     checkpoint = torch.load(path, map_location = 'cpu')['model']
    #     model.load_state_dict(checkpoint)
    #     # print(model)
    #     # model = torch.load(path, map_location = 'cpu')
    #     model.cuda()
    #     model.eval()
    #     logit = []
    #     targets = []
    #     with torch.no_grad():
    #         for idx, (input, target) in enumerate(val_dataloader):
    #             input = input.cuda()
    #             logits_mask = model(input)
    #             logit.append(logits_mask)
    #             targets.append(target)
    #         logit_ = torch.cat(logit, dim = 0)
    #         target_ = torch.cat(targets, dim = 0)
    #     result = {
    #         'logit': logit_,
    #         'target': target_
    #     }
    #     with open(f'best_result/{key}.pkl', 'wb') as file:
    #         pickle.dump(result, file)
    #     print(f'{key} saved')


    pkl_list = {
        'best_val_model_1': 'final_result/best_test_model_1.pkl',
        'best_val_model_2': 'final_result/best_test_model_2.pkl',
        'best_val_model_3': 'final_result/best_test_model_3.pkl',
        'best_val_model_4': 'final_result/best_test_model_4.pkl',
        'best_val_model_5': 'final_result/best_test_model_5.pkl',
        'best_val_model_6': 'final_result/best_test_model_6.pkl',
        'best_val_model_7': 'final_result/best_test_model_7.pkl',

    #     # 'best_val_model_1': 'best_result/best_val_model_1.pkl',
    #     # 'best_val_model_2': 'best_result/best_val_model_2.pkl',
    #     # 'best_val_model_3': 'best_result/best_val_model_3.pkl',
    #     # 'best_val_model_4': 'best_result/best_val_model_4.pkl',
    #     # 'best_val_model_5': 'best_result/best_val_model_5.pkl',
    #     # 'best_val_model_6': 'best_result/best_val_model_6.pkl',

    #     # 'best_val_model_1': 'test_label/best_val_model_1.pkl',
    #     # 'best_val_model_2': 'test_label/best_val_model_2.pkl',
    #     # 'best_val_model_3': 'test_label/best_val_model_3.pkl',
    #     # 'best_val_model_4': 'test_label/best_val_model_4.pkl',
    #     # 'best_val_model_5': 'test_label/best_val_model_5.pkl',
    #     # 'best_val_model_6': 'test_label/best_val_model_6.pkl',

    #     # 'model_1': 'best_result/model_1.pkl',
    #     # 'model_2': 'best_result/model_2.pkl',
    #     # 'model_3': 'best_result/model_3.pkl',
    #     # 'model_4': 'best_result/model_4.pkl',
    #     # 'model_5': 'best_result/model_5.pkl',
    #     # 'model_6': 'best_result/model_6.pkl',
    #     # 'model_7': 'best_result/model_7.pkl',
    #     # 'model_8': 'test_label/model_8.pkl',
    #     # 'model_11': 'test_label/model_11.pkl',
    #     # 'model_12': 'test_label/model_12.pkl',
    #     # 'model_13': 'test_label/model_13.pkl',
    #     # 'model_9': 'best_result/model_9.pkl',
    #     # 'model_10': 'best_result/model_10.pkl',

    }


    # test label
    potential_logit = []
    for key, path in pkl_list.items():
        with open(path, 'rb') as f:
            results = pickle.load(f)
        logit = results['logit'].cuda()
        potential_logit.append(logit)
    ensemble_output = ensemble(potential_logit, 0.5, 0.5, 1).detach().cpu()
    with open(f'final_result/test_ensemble_1.pkl', 'wb') as file:
        pickle.dump(ensemble_output, file)



    # potential_logit = []
    # for key, path in pkl_list.items():
    #     with open(path, 'rb') as f:
    #         results = pickle.load(f)
    #     logit = results['logit'].cuda()
    #     target = results['target']
    #     potential_logit.append(logit)
    # ensemble_output = ensemble(potential_logit, 0.5, 0.5, 1).detach().cpu()
    # tp, fp, fn, tn = smp.metrics.get_stats(ensemble_output.long(), target.long(), mode="binary")
    # dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    # print(dataset_iou)

    # # # # greedy ensemble
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
    # final_ensemble_result = []
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
    #         final_ensemble_result = copy.copy(ensemble_output)

    # with open(f'best_result/val_ensemble.pkl', 'wb') as file:
    #     pickle.dump(final_ensemble_result, file)
    # with open('ensemble_result.csv', 'a') as f:
    #     f.write(f'{ensemble_mode}\t{tensor_list}\t{iou}\n')  
