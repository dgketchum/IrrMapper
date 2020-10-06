import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms
from webdataset import dataset as wds

from data_prep.pixel_preparation import BANDS, CHANNELS, DATES

SEQUENCE_LENGTH = len(DATES.keys())

structure = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])


def transform_(x, mean_std):
    normalize = transforms.Normalize(
        mean=mean_std[0],
        std=mean_std[1])
    transform = transforms.Compose([normalize])
    x = transform(x)
    return x


def pixel_dataset(mode, config, norm):
    def map_fn(item):
        item = item['pth']
        x = item[:, :, :BANDS]
        x = x.reshape((x.shape[0] * x.shape[1], SEQUENCE_LENGTH, CHANNELS))
        y = item[:, :, 98:]
        y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])
        mask = y.sum(1) > 0
        print(y.sum())
        y = y[mask]
        print(y.shape)
        if y.shape[0] > 0:
            y = y.argmax(1)
            x = x[mask]
            x = transform_(x, norm)
            return x, y
        else:
            return None

    root = config['dataset_folder']
    loc = os.path.join(root, mode, '{}_patches'.format(mode))
    end_idx = len(os.listdir(loc)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(loc, brace_str)
    dataset = wds.Dataset(url).decode('torchl')  # .map(map_fn)
    for i, s in enumerate(dataset):
        lab = s['pth'][:, :, 101].numpy()
        samples = lab.sum().item()
        if samples == 0:
            label = s['pth'][:, :, 98:].numpy()
            print(i, label)
    return dataset


class PixelDataChunk(data.Dataset):
    def __init__(self, _file, labels, norm=None, extra_feature=None):

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
