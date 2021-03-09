import os
import torch
import webdataset as wds
from webdataset import dataset as wds

from configure import BANDS, CHANNELS, TERRAIN, SEQUENCE_LEN


def find_archives(path):
    tar_list = []
    for top_dir, dir_list, obj_list in os.walk(path):
        tar_list.extend([os.path.join(top_dir, obj) for obj in obj_list if obj.endswith('.tar')])
    return tar_list


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
        y -= 1
        if config['model'] == 'nnet':
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        if config['model'] == 'tcnn':
            x = x.permute(0, 2, 1)
        return x, y, g

    root = config['dataset_folder']
    loc = os.path.join(root, mode)
    urls = find_archives(loc)
    ds = wds.Dataset(urls)
    ds = ds.decode('torchl').map(map_input).batched(config['batch_size'])
    return ds


def temporal_image_dataset(mode, config, norm):
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
    urls = find_archives(loc)
    dataset = wds.Dataset(urls).decode('torchl').map(map_fn).batched(config['batch_size'])
    return dataset


def image_dataset(mode, config, norm):
    """ Use for training and prediction of image datasets"""

    def map_fn(item):
        features = item['pth'][:, :, :BANDS + TERRAIN]
        # features = transform_(features, norm).float()
        features = features.float()
        x = features[:, :, :BANDS]
        x = x.permute((2, 0, 1)).float()
        y = item['pth'][:, :, -1].long()
        g = features[:, :, BANDS:BANDS + TERRAIN].permute(2, 0, 1)
        return x, y, g

    data_dir = config['dataset_folder']
    loc = os.path.join(data_dir, mode)
    urls = find_archives(loc)
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
    urls = find_archives(loc)
    dataset = wds.Dataset(urls).decode('torchl').map(map_fn).batched(config['batch_size'])
    return dataset
