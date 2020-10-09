import os
import pickle as pkl
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from webdataset import dataset as wds

from data_prep import BANDS, CHANNELS, SEQUENCE_LEN


def transform_(x, mean_std):
    normalize = transforms.Normalize(
        mean=mean_std[0].reshape(mean_std[0].shape[0] * mean_std[0].shape[1]),
        std=mean_std[1].reshape(mean_std[0].shape[0] * mean_std[0].shape[1]))
    t = transforms.Compose([normalize])
    x = t(x)
    return x


def image_dataset(mode, data_dir, norm):
    def map_fn(item):
        item = item['pth']
        x = item[:, :, :BANDS]
        x = x.permute(2, 0, 1)
        x = transform_(x, norm)
        x = x.reshape(x.shape[1], x.shape[2], SEQUENCE_LEN, CHANNELS)
        x = x.permute((2, 3, 0, 1)).float()
        y = item[:, :, -4:].permute(2, 0, 1).int()
        return x, y

    loc = os.path.join(data_dir, '{}_patches'.format(mode))
    end_idx = len(os.listdir(loc)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(loc, brace_str)
    dataset = wds.Dataset(url).decode('torchl').map(map_fn)
    return dataset


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
