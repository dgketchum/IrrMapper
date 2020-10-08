import os
import numpy as np
import json
import torch
from torch.utils import data
from torchvision import transforms
from webdataset import dataset as wds

from data_prep.pixel_preparation import CHANNELS, DATES

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
    dataset = wds.Dataset(url).decode('torchl').to_tuple('pth', 'json').map_tuple(map_input, identity).batched(1)
    return dataset


def pixel_data(mode, config, norm, extra_feature=False):
    root = config['dataset_folder']
    labels = config['nomenclature']
    loc = os.path.join(root, mode, '{}_patches'.format(mode))
    l = [os.path.join(loc, f) for f in os.listdir(loc) if f.endswith('labels.npy')]
    dl = [os.path.join(loc, f) for f in os.listdir(loc) if
          os.path.join(loc, f) not in l and f.endswith('.npy')]
    datasets = []
    for f in dl:
        pds = PixelDataChunk(f, labels=labels,
                             norm=norm,
                             extra_feature=extra_feature)
        datasets.append(pds)
    dt = data.ConcatDataset(datasets)
    return dt


class PixelDataChunk(data.Dataset):
    def __init__(self, _file, labels, norm=None, extra_feature=False):

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

        if self.extra_feature:
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

        m, s = self.norm
        m = np.array(m)
        s = np.array(s)
        x = (x - m) / s

        x = x.astype('float')
        data = torch.from_numpy(x).float()

        if self.extra_feature:
            ef = (self.extra[str(item)] - self.extra_m) / self.extra_s
            ef = torch.from_numpy(ef).float()

            ef = torch.stack([ef for _ in range(data[0].shape[0])], dim=0)
            data = (data, ef)

        return data, torch.from_numpy(np.array(y, dtype=int))
