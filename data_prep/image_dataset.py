
import os

import numpy as np

import torch
from torch.utils import data


class ImageDataset(data.Dataset):
    def __init__(self, _dir, norm):

        super(ImageDataset, self).__init__()

        self.data_src = _dir
        self.norm = norm

        self.img_list = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.npy')]
        self.len = len(self.img_list)

    def __len__(self):
        return self.len

    def __getitem__(self, item):

        a = np.load(self.img_list[item])
        x = a[:, :, :-1]
        y = a[:, :, -1:]

        if self.norm is not None:
            m, s = self.norm
            m = np.array(m)
            s = np.array(s)
            x = (x - m) / s

        data = torch.from_numpy(x).float()

        return data, torch.from_numpy(np.array(y, dtype=int))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
