import os
from itertools import islice
import numpy as np
from subprocess import Popen
import torch
from torch.utils import data
from webdataset import dataset as wds
from webdataset import MultiDataset as wmds

from data_prep.pixel_preparation import BANDS, CHANNELS, DATES
from data_prep.bucket import get_bucket_contents

def image_dataset(mode):
    bucket = get_bucket_contents('ts_data')
    gs_dir = 'cmask/tar/{}/{}_patches'.format(mode, mode)
    tar_data = bucket[gs_dir]
    url = 'http://storage.googleapis.com/ts_data/cmask/tar/train/train_patches/train_000001.tar'
    url = f"pipe: curl -s -L {url} || true"
    bs = 20
    dataset = wds.Dataset(url).decode('torchl').shuffle(100)
    for sample in islice(dataset, 0, 3):
        for key, value in sample.items():
            print(key, repr(value)[:50])
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
    image_dataset('train')
# ========================= EOF ====================================================================
