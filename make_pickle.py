import pickle
import torch
import numpy as np
# with open('./dataset_pickle_2/dataset_pickle/train/fc_post_matrix.pkl', 'rb') as file:
#         train_fc_post_matrix = pickle.load(file)

# with open('./dataset_pickle_2/dataset_pickle/val/fc_post_matrix_folder0.pkl', 'rb') as file:
#         val_fc_post_matrix = pickle.load(file)

# with open('./dataset_pickle_2/dataset_pickle/train/train_mask.pkl', 'rb') as file:
#         train_mask = pickle.load(file)

# with open('./dataset_pickle_2/dataset_pickle/val/val_mask.pkl', 'rb') as file:
#         val_mask = pickle.load(file)

with open('./dataset_pickle_2/dataset_pickle/test/test_mask_new_2.pkl', 'rb') as file:
        test_mask = pickle.load(file)
print(test_mask.shape)
test_mask = np.transpose(test_mask, (0, 2, 3, 1))
print(test_mask.shape)
# train_val_fc_post_matrix = np.concatenate((train_fc_post_matrix, val_fc_post_matrix), axis=0)
# train_val_mask = np.concatenate((train_mask, val_mask), axis=0)

# print(train_fc_post_matrix.shape)
# print(val_fc_post_matrix.shape)
# print(train_val_fc_post_matrix.shape)

# print(train_mask.shape)
# print(val_mask.shape)
# print(train_val_mask.shape)

# with open('./dataset_pickle_2/dataset_pickle/train/train_val_fc_post_matrix.pkl', 'wb') as file:
#         pickle.dump(train_val_fc_post_matrix, file)

# with open('./dataset_pickle_2/dataset_pickle/train/train_val_mask.pkl', 'wb') as file:
#         pickle.dump(train_val_mask, file)

with open('./dataset_pickle_2/dataset_pickle/test/test_mask_2.pkl', 'wb') as file:
        pickle.dump(test_mask, file)