import os
import pickle as pkl
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from webdataset import dataset as wds

from data_prep.pixel_preparation import BANDS, CHANNELS, DATES

sequence_len = len(DATES.keys())


def transform_(x, mean_std):
    normalize = transforms.Normalize(
        mean=mean_std[0].reshape(mean_std[0].shape[0] * mean_std[0].shape[1]),
        std=mean_std[1].reshape(mean_std[0].shape[0] * mean_std[0].shape[1]))
    transform_ = transforms.Compose([normalize])
    x = transform_(x)
    return x


def image_dataset(mode, config, norm):

    def map_fn(item):
        item = item['pth']
        x = item[:, :, :BANDS]
        x = x.permute(2, 0, 1)
        x = transform_(x, norm)
        x = x.reshape(x.shape[1], x.shape[2], len(DATES.keys()), CHANNELS)
        x = x.permute((2, 3, 0, 1)).float()
        y = item[:, :, -4:].permute(2, 0, 1).int()
        return x, y

    root = config['dataset_folder']
    loc = os.path.join(root, mode, '{}_patches'.format(mode))
    end_idx = len(os.listdir(loc)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(loc, brace_str)
    dataset = wds.Dataset(url).decode('torchl').map(map_fn)
    return dataset


class ImageDataset(data.Dataset):
    def __init__(self, _dir, norm):
        super(ImageDataset, self).__init__()

        self.data_src = _dir
        self.norm = norm
        self.sequence_len = len(DATES.keys())
        self.channels = CHANNELS
        self.bands = BANDS

        self.img_list = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.npy')]
        self.len = len(self.img_list)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        a = np.load(self.img_list[item])

        # needs B, T, C, H, W
        x = a[:, :, :self.bands]
        x = x.reshape(x.shape[0], x.shape[1], self.sequence_len, self.channels)
        x = x.transpose((2, 3, 0, 1))
        # if torch.any(x) == -1.:
        #     raise TypeError('found missing value')

        y = a[:, :, -4:]
        y = torch.from_numpy(y).permute(2, 0, 1)

        if self.norm is not None:
            m, s = self.norm
            m = np.array(m).reshape((self.sequence_len, self.channels, 1, 1))
            s = np.array(s).reshape((self.sequence_len, self.channels, 1, 1))
            x = (x - m) / s

        x = torch.from_numpy(x).float()
        return x, y


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
