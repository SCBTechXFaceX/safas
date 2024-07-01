import os, torch
from utils import protocol_decoder
import math


def get_single_dataset(data_dir, FaceDataset, data_name="", train=True, label=None, transform=None, UUID=-1):
    if train:
        data_set = FaceDataset(data_name, os.path.join(data_dir, data_name), split='train', label=label, transform=transform,  UUID=UUID)
    else:
        data_set = FaceDataset(data_name, os.path.join(data_dir, data_name), split='test', label=label,
                                      transform=transform, UUID=UUID)
    
    # print("Loading {}, number: {}".format(data_name, len(data_set)))
    return data_set

def get_datasets(data_dir, FaceDataset, train=True, protocol="1", transform=None):

    data_name_list_train, data_name_list_test = protocol_decoder(protocol)

    sum_n = 0
    if train:
        data_set_sum = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[0], train=True, transform=transform, UUID=0)
        sum_n = len(data_set_sum)
        for i in range(1, len(data_name_list_train)):
            data_tmp = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[i], train=True, transform=transform, UUID=i)
            data_set_sum += data_tmp
            sum_n += len(data_tmp)
    else:
        data_set_sum = {}
        for i in range(len(data_name_list_test)):
            data_tmp = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_test[i], train=False, transform=transform, UUID=i)
            data_set_sum[data_name_list_test[i]] = data_tmp
            sum_n += len(data_tmp)
    print("Total number: {}".format(sum_n))
    return data_set_sum
