import os
import numpy as np
import torch
from webdataset import dataset as wds

from data_prep import CHANNELS, DATES

SEQUENCE_LENGTH = len(DATES.keys())

structure = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])


def identity(y):
    y = y['labels']
    y = torch.tensor(y)
    return y


def transform_(x, mean_std):
    x = (x - mean_std[0]) / mean_std[1]
    return x


def pixel_dataset(mode, config, norm, extra_feature=False):
    def map_input(x):
        x = x.permute(2, 0, 1)
        x = transform_(x, norm)
        return x.float()

    root = config['dataset_folder']
    loc = os.path.join(root, mode, '{}_patches'.format(mode))
    end_idx = len(os.listdir(loc)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(loc, brace_str)
    ds = wds.Dataset(url)
    ds = ds.decode('torchl').to_tuple('pth', 'json')
    ds = ds.map_tuple(map_input, identity).batched(1)
    return ds

