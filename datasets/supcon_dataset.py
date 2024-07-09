import os

import PIL.Image
import torch
import pandas as pd
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import math
from glob import glob
import re
from utils.rotate_crop import crop_rotated_rectangle, inside_rect, vis_rotcrop
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt

from ylib.scipy_misc import imread, imsave
from .meta import DEVICE_INFOS

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

class FaceDataset(Dataset):
    def __init__(self, dataset_name, root_dir, split='train', transform=None, UUID=-1, model_name='safas'):
        self.split = split
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transform = transform
        self.UUID = UUID
        self.df = pd.read_csv(os.path.join(root_dir, 'label.csv'))
        self.safas = model_name

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        image_path = os.path.join(self.root_dir, self.df.iloc[idx]['file_path'])
        
        if self.split == 'train':
            image_x = imread(image_path)
            image_x_view1 = self.transform(PIL.Image.fromarray(image_x))
            image_x_view2 = self.transform(PIL.Image.fromarray(image_x))
        else:
            image_x = imread(image_path)
            image_x_view1 = self.transform(PIL.Image.fromarray(image_x))
            image_x_view2 = image_x_view1
            
        if self.df.iloc[idx]['label'] == 'spoof':
            label = 1
        else:
            label = 0

        if self.safas == 'safas':
            sample = {"image_x_v1": np.array(image_x_view1),
                    "image_x_v2": np.array(image_x_view2),
                    "label": label,
                    "UUID": self.UUID}
            return sample
        else:
            return image_x_view1, label
        
        

