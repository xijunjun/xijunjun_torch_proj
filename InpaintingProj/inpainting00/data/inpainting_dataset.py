import os.path
from os.path import join
from data.image_folder import make_dataset
from data.transforms import Sobel, to_norm_tensor, to_tensor, ReflectionSythesis_1, ReflectionSythesis_2
from PIL import Image
import random
import torch
import math

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import util.util as util
import data.torchdata as torchdata
import  cv2

BaseDataset = torchdata.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()

class InpaintingDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None):
        super(InpaintingDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir))
        # print(self.fns)

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        B = -1

        # print(join(self.datadir, fn))

        # m_img = Image.open(join(self.datadir, fn)).convert('RGB')

        m_img=cv2.imread(join(self.datadir, fn))

        m_img=cv2.resize(m_img,(512,512))

        # cv2.imshow('m_img',m_img)
        # key=cv2.waitKey(0)

        M=m_img
        M = to_tensor(m_img)
        B=M

        # data['input'], data['target_t'], data['target_r']

        data = {'input': M, 'target_t': B, 'target_r':M,'fn': fn}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)










