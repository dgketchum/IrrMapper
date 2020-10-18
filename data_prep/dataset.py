import os
from glob import glob, iglob
import torch
import webdataset as wds
from webdataset import dataset as wds

from data_prep import BANDS, CHANNELS, TERRAIN, SEQUENCE_LEN


def find_archives(path):
    for file in iglob(path, recursive=True):
        [os.path.join(loc, x) for x in os.listdir(loc) if x.endswith('.tar')]
        print(file)
    return file


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
    loc = os.path.join(root, mode)
    urls = find_archives(loc)
    ds = wds.Dataset(urls)
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
    loc = os.path.join(root, mode)
    urls = [os.path.join(loc, x) for x in os.listdir(loc) if x.endswith('.tar')]
    ds = wds.Dataset(urls)
    ds = ds.decode('torchl').map(map_input).batched(config['batch_size'])
    return ds


def image_dataset(mode, config, norm):
    """ Use for training and prediction of image datasets"""
    def map_fn(item):
        features = item['pth'][:, :, :BANDS + TERRAIN]
        features = transform_(features, norm).float()
        x = features[:, :, :BANDS]
        x = x.reshape(x.shape[0], x.shape[1], SEQUENCE_LEN, CHANNELS)
        x = x.permute((2, 3, 0, 1)).float()
        y = item['pth'][:, :, -4:].permute(2, 0, 1).int()
        g = features[:, :, BANDS:BANDS + TERRAIN].permute(2, 0, 1)
        return x, y, g

    data_dir = config['dataset_folder']
    loc = os.path.join(data_dir, mode)
    urls = [os.path.join(loc, x) for x in os.listdir(loc) if x.endswith('.tar')]
    dataset = wds.Dataset(urls).decode('torchl').map(map_fn).batched(config['batch_size'])
    return dataset


def predict_dataset(config, norm):
    """ Use for prediction of images using pixel-based models """
    def map_fn(item):
        features = item['pth'][:, :, :BANDS + TERRAIN]
        features = transform_(features, norm).float()
        x = features[:, :, :BANDS]
        x = x.reshape(x.shape[0], x.shape[1], SEQUENCE_LEN, CHANNELS).float()
        y = item['pth'][:, :, -4:].permute(2, 0, 1)
        g = features[:, :, BANDS:BANDS + TERRAIN].permute(2, 0, 1)
        return x, y, g

    data_dir = config['prediction_dir']
    loc = os.path.join(data_dir)
    urls = [os.path.join(loc, x) for x in os.listdir(loc) if x.endswith('.tar')]
    dataset = wds.Dataset(urls).decode('torchl').map(map_fn).batched(config['batch_size'])
    return dataset