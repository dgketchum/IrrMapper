"""
Credits to  https://github.com/VSainteuf/pytorch-psetae
author: Vivien Sainte Fare Garnot
paper: https://arxiv.org/pdf/2007.00586.pdf
"""
import torch
from torch.utils import data

import pandas as pd
import numpy as np

import os
import json
from pathlib import Path


def pixel_data(folder, labels, norm, extra_feature):
    folder = os.path.join(folder, 'DATA')
    l = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('labels.npy')]
    dl = [os.path.join(folder, f) for f in os.listdir(folder) if
          os.path.join(folder, f) not in l and f.endswith('.npy')]
    datasets = []
    for f in dl:
        pds = PixelDataChunk(f, labels=labels,
                             norm=norm,
                             extra_feature=extra_feature)
        datasets.append(pds)
    dt = data.ConcatDataset(datasets)
    return dt


class PixelDataChunk(data.Dataset):
    def __init__(self, _file, labels, norm=None, extra_feature=None):
        """

        Args:
            folder (str): path to the main folder of the dataset, formatted as indicated in the readme
            labels (str): name of the nomenclature to use in the labels.json file
            sub_classes (list): If provided, only the samples from the given list of classes are considered.
            (Can be used to remove classes with too few samples)
            norm (tuple): (mean,std) tuple to use for normalization
            extra_feature (str): name of the additional static feature file to use
            jitter (tuple): if provided (sigma, clip) values for the addition random gaussian noise
            return_id (bool): if True, the id of the yielded item is also returned (useful for inference)
        """
        super(PixelDataChunk, self).__init__()

        self.data_src = _file
        self.label_src = _file.replace('.npy', '_labels.npy')
        self.folder = Path(os.path.dirname(_file)).parents[0]
        self.data_folder = os.path.join(self.folder, 'DATA')
        self.meta_folder = os.path.join(self.folder, 'META')
        self.labels = labels
        self.norm = norm

        self.extra_feature = extra_feature

        self.data = np.load(self.data_src)
        self.target = list(np.load(self.label_src))
        self.len = len(self.target)

        if self.extra_feature is not None:
            with open(os.path.join(self.meta_folder, '{}.json'.format(extra_feature)), 'r') as file:
                self.extra = json.loads(file.read())

            if isinstance(self.extra[list(self.extra.keys())[0]], int):
                for k in self.extra.keys():
                    self.extra[k] = [self.extra[k]]
            df = pd.DataFrame(self.extra).transpose()
            self.extra_m, self.extra_s = np.array(df.mean(axis=0)), np.array(df.std(axis=0))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        """
        Returns a Pixel-Set sequence tensor with its pixel mask and optional additional features.
        For each item npixel pixels are randomly dranw from the available pixels.
        If the total number of pixel is too small one arbitrary pixel is repeated. The pixel mask keeps track of true
        and repeated pixels.
        Returns:
              (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features) with:
                Pixel-Set: Sequence_length x Channels x npixel
                Pixel-Mask : Sequence_length x npixel
                Extra-features : Sequence_length x Number of additional features

        """
        x = self.data[:, :, item]
        y = self.target[item]

        if self.norm is not None:
            m, s = self.norm
            m = np.array(m)
            s = np.array(s)
            x = (x - m) / s

        x = x.astype('float')
        data = torch.from_numpy(x).float()

        if self.extra_feature is not None:
            ef = (self.extra[str(item)] - self.extra_m) / self.extra_s
            ef = torch.from_numpy(ef).float()

            ef = torch.stack([ef for _ in range(data[0].shape[0])], dim=0)
            data = (data, ef)

        return data, torch.from_numpy(np.array(y, dtype=int))


