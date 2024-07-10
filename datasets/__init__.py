import os, torch
import math


def get_single_dataset(data_dir, FaceDataset, data_name="", train=True, transform=None, UUID=-1, model_name='safas'):
    if train:
        data_set = FaceDataset(data_name, os.path.join(data_dir, data_name), split='train', transform=transform,  UUID=UUID, model_name=model_name)
    else:
        data_set = FaceDataset(data_name, os.path.join(data_dir, data_name), split='test',
                                      transform=transform, UUID=UUID, model_name=model_name)
    
    # print("Loading {}, number: {}".format(data_name, len(data_set)))
    return data_set

def get_datasets(data_dir, FaceDataset, train=True, target="1", transform=None, model_name='safas'):
    data_folder = os.listdir(data_dir)
    data_name_list_train = []
    for folder in data_folder:
        if folder != target:
            data_name_list_train = [folder]
    sum_n = 0
    if train:
        data_set_sum = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[0], train=True, transform=transform, UUID=0, model_name=model_name)
        sum_n = len(data_set_sum)
        for i in range(1, len(data_name_list_train)):
            data_tmp = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[i], train=True, transform=transform, UUID=i, model_name=model_name)
            data_set_sum += data_tmp
            sum_n += len(data_tmp)
    else:
        data_set_sum = {}
        data_tmp = get_single_dataset(data_dir, FaceDataset, data_name=target, train=False, transform=transform, UUID=0, model_name=model_name)
        data_set_sum[target] = data_tmp
        sum_n += len(data_tmp)
    print("Total number: {}".format(sum_n))
    return data_set_sum
