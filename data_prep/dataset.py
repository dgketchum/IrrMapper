import os
import numpy as np
import torch
import webdataset as wds
from webdataset import dataset as wds

from data_prep import BANDS, CHANNELS, TERRAIN, SEQUENCE_LEN

structure = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])


def transform_(x, mean_std):
    x = (x - mean_std[0]) / mean_std[1]
    return x


def pixel_dataset(mode, config, norm):
    def map_input(item):
        features = item['pth'][:, :BANDS + TERRAIN]
        features = transform_(features, norm).float()
        x, g = features[:, :BANDS], features[:, BANDS:BANDS + TERRAIN]
        x = x.reshape((x.shape[0], SEQUENCE_LEN, CHANNELS))
        y = item['pth'][:, -1].long()
        return x, y, g

    root = config['dataset_folder']
    loc = os.path.join(root, mode, '{}_patches'.format(mode))
    end_idx = len(os.listdir(loc)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(loc, brace_str)
    ds = wds.Dataset(url)
    ds = ds.decode('torchl').map(map_input).batched(config['batch_size'])
    return ds


def pixelset_dataset(mode, config, norm):
    n_pixel = config['n_pixel']

    def map_input(item):
        features = item['pth'][:, :BANDS + TERRAIN]
        features = transform_(features, norm).float()
        x, g = features[:, :BANDS], features[:, BANDS:BANDS + TERRAIN]
        x = x.reshape((x.shape[0], SEQUENCE_LEN, CHANNELS))
        y = item['pth'][0, -1].long()

        if x.shape[0] >= n_pixel:
            perm = torch.randperm(x.shape[0])
            idx = perm[:n_pixel]
            x = x[idx, :, :]
            g = g[idx, :]
        else:
            repeats = torch.ceil(torch.tensor(n_pixel).float() / x.shape[0]).int()
            x = x.repeat(repeats, 1, 1)[:n_pixel, :, :]
            g = g.repeat(repeats, 1)[:n_pixel, :]
        x = x.permute(1, 2, 0)
        return x, y, g

    root = config['dataset_folder']
    loc = os.path.join(root, mode, '{}_patches'.format(mode))
    end_idx = len(os.listdir(loc)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(loc, brace_str)
    ds = wds.Dataset(url)
    ds = ds.decode('torchl').map(map_input).batched(config['batch_size'])
    return ds


def image_dataset(mode, config, norm, pixel_predict=False):

    def map_fn(item):
        features = item['pth'][:, :, :BANDS + TERRAIN]
        features = transform_(features, norm).float()
        x = features[:, :, :BANDS]
        x = x.permute(2, 0, 1)

        if pixel_predict:
            x = x.permute(1, 2, 0)
            x = x.reshape(x.shape[0], x.shape[1], SEQUENCE_LEN, CHANNELS).float()
            y = item['pth'][:, :, -4:].permute(2, 0, 1)
        else:
            x = x.reshape(x.shape[1], x.shape[2], SEQUENCE_LEN, CHANNELS)
            x = x.permute((2, 3, 0, 1)).float()
            y = item['pth'][:, :, -4:].permute(2, 0, 1).int()

        g = features[:, :, BANDS:BANDS + TERRAIN].permute(2, 0, 1)
        return x, y, g

    if pixel_predict:
        data_dir = config['prediction_dir']
        loc = os.path.join(data_dir, '{}_patches'.format(mode))
    else:
        data_dir = config['dataset_folder']
        loc = os.path.join(data_dir, mode, '{}_patches'.format(mode))

    end_idx = len(os.listdir(loc)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(loc, brace_str)
    dataset = wds.Dataset(url).decode('torchl').map(map_fn).batched(config['batch_size'])
    return dataset