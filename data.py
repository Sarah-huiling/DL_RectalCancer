from __future__ import print_function

import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import csv
import xlrd


def load_label_form_xl(xl_path):
    reads = xlrd.open_workbook(xl_path)
    label = []
    for row in range(reads.sheet_by_index(0).nrows):
        label.append(reads.sheet_by_index(0).cell(row, 0).value)
        label.append(reads.sheet_by_index(0).cell(row, 1).value)
    return label


def load_npy(x_path):
    label = [0, 0]
    data = np.load(x_path, allow_pickle=True)

    labels = load_label_form_xl('/media/zhl/ResearchData/20221028HX-rectalCancer/labels.xls')
    id_index = labels.index(x_path.split('/')[-1])  # 提取PID，
    # id_index = labels.index(x_path.split('/')[-1][:-4])

    if labels[id_index+1] == 1.0:
        label = [0, 1]
    if labels[id_index+1] == 0.0:
        label = [1, 0]
    return data, label


class MyDataset(Dataset):
    def __init__(self, root, transform=True):
        self.image_files = np.array(root)
        self.transform = transform
        # self.transform = transform

    def __getitem__(self, index):  # 返回的是tensor

        x, y = load_npy(self.image_files[index])
        x = x.astype(np.int16)
        return torch.FloatTensor(x), torch.FloatTensor(y)

    def __len__(self):
        return len(self.image_files)

    def transform(x):
        # im_aug = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()])
        patch_size = (128, 128, 128)
        im_aug = get_train_transform(patch_size)
        x = im_aug(x)
        return x


# cv2.imshow('2',dataset.images[0,:,:,:])
# cv2.waitKey(0)
