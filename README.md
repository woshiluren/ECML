### Requirements
segmentation_models_pytorch==0.3.3
torch

### Project Directory
+ final_model: Pre-trained model weight files.
+ final_result: 
    + final_result/best_test_model_i.pkl: Ensemble candidate model outputs.
    + final_result/test_ensemble.pkl: The ensemble outputs.
+ dataset_pickle: Processed data.
    + dataset_pickle/test: Test Set.
    + dataset_pickle/val: Validation Set.
    + dataset_pickle/train: Train Set.
+ configs: The configs for model preparation.
+ aug_config.json: Setting up the file for data augmentation.
+ ensemble.py: Run it to ensemble the candidate models.
+ utils.py: Running auxiliary functions in the code.
+ data_aug.py: The data augmentation.
+ unimatch.py: Transductive learning method.
+ semi_main.py: Training models.


### Training Model
Train a model with transductive learning.
```bash
python unimatch.py --gpu YOUR_GPU
```

Train a model without transductive learning.
```bash
python train.py --configs configs/DeepLab_plus_resnet152.yaml --lr 0.0001 --lamda 0 --max-epoch 50 
```

### Evaluate Models
Use pretrained models to generate the output(e.g. final_result\best_test_model_1.pkl) and ensemble the results of those models(e.g.final_result\test_ensemble_1.pkl).
```bash
python ensemble.py 
```

Convert the ensemble result (e.g. final_result\test_ensemble_1.pkl) to final submission file(e.g. result_to_submit\final_predictions_1.csv).
```bash
python create_sample_submission.py
```