import os
from glob import glob
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from configure import BANDS, CHANNELS, TERRAIN, SEQUENCE_LEN
from data_load import NORM

SELECT_CHANNELS = torch.tensor([44, 45, 48, 65, 66, 69])

"""WARNING This dataloader hijacks 'test' data and puts it in the training data stack, when test_loader is called
    it is just getting the valid data. For pixel models, it pulls valid data in image format, rather than pixel stack"""

# only sample-wise normalization is for tcnn, others use model


class IrrMapDataset(Dataset):

    def __init__(self, data_dir, mode, transforms=None, model=None):
        self.data_dir = data_dir
        self.mode = mode
        self.model = model

        try:
            self.mean = torch.tensor(NORM[0][:91]).float()
            self.std = torch.tensor(NORM[1][:91]).float()
        except TypeError:
            pass

        self.img_paths = glob(os.path.join(data_dir, '*.pth'))
        # add test to train for two-way train/valid split
        if 'train' in data_dir:
            add_paths = glob(os.path.join(data_dir.replace('train', 'test'), '*.pth'))
            self.img_paths += add_paths

        self.transforms = transforms

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img = torch.load(img_path)

        if self.mode == 'pixel':
            img = img[torch.randperm(img.size()[0])]
            features = img[:, :BANDS + TERRAIN]
            x, g = features[:, :BANDS], features[:, BANDS:BANDS + TERRAIN]
            y = img[:, -1]

            if self.model == 'nnet':
                pass

            if self.model == 'tcnn':
                x = self.transform_(x)
                x = x.reshape((x.shape[0], CHANNELS, SEQUENCE_LEN))

        elif self.mode == 'image':
            features = img[:, :, :BANDS + TERRAIN]
            x = features[:, :, :BANDS]
            x = x.permute((2, 0, 1))
            y = img[:, :, -1]
            g = features[:, :, BANDS:BANDS + TERRAIN].permute(2, 0, 1)

            if self.model == 'nnet':
                x = x.reshape(x.shape[1] * x.shape[2], x.shape[0])

            if self.model == 'tcnn':
                x = x.reshape(x.shape[-1] * x.shape[-1], CHANNELS * SEQUENCE_LEN)
                x = self.transform_(x)
                x = x.reshape(x.shape[0], CHANNELS, SEQUENCE_LEN)

        elif self.mode == 'temporal_image':
            features = img[:, :, :BANDS + TERRAIN]
            x = features[:, :, :BANDS]
            x = x.reshape(x.shape[0], x.shape[1], SEQUENCE_LEN, CHANNELS)
            x = x.permute((2, 3, 0, 1))
            y = img[:, :, -4:].permute(2, 0, 1).int()
            g = features[:, :, BANDS:BANDS + TERRAIN].permute(2, 0, 1)

        else:
            raise KeyError(
                'Must choose from {} image modes'.format(['pixel',
                                                          'image',
                                                          'temporal_image']))

        if not torch.isfinite(img).all():
            print('bad image {}'.format(self.img_paths[item]))
            x, g, y = self.__getitem__(item + 1)

        return x.float(), g.float(), y.long()

    def __len__(self):
        return len(self.img_paths)

    def transform_(self, x):
        x = (x - self.mean) / (self.std + 1e-6)
        return x


class IrrMapDataModule(pl.LightningDataModule):

    def __init__(self, config):
        pl.LightningDataModule.__init__(self, config)
        self.num_workers = config['num_workers']
        self.data_dir = os.path.join(config['dataset_folder'])
        self.batch_sz = config['batch_size']
        self.mode = config['mode']
        self.model = config['model']

    def train_dataloader(self):
        train_dir = os.path.join(self.data_dir, 'train')
        train_ds = IrrMapDataset(train_dir, self.mode, transforms=None, model=self.model)
        dl = DataLoader(
            train_ds,
            shuffle=True,
            batch_size=self.batch_sz,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.collate_fn)
        return dl

    def val_loader(self):
        val_dir = os.path.join(self.data_dir, 'valid')
        valid_ds = IrrMapDataset(val_dir, self.mode, transforms=None, model=self.model)
        dl = DataLoader(
            valid_ds,
            shuffle=False,
            batch_size=self.batch_sz,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn)
        return dl

    def test_loader(self):
        test_dir = os.path.join(self.data_dir, 'test')
        test_ds = IrrMapDataset(test_dir, self.mode, transforms=None, model=self.model)
        dl = DataLoader(
            test_ds,
            shuffle=False,
            batch_size=self.batch_sz,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn)
        return dl

    @staticmethod
    def collate_fn(data):
        x, g, y = [], [], []
        for d in data:
            if d:
                x.append(d[0]), g.append(d[1]), y.append(d[2])
        return torch.stack(x), torch.stack(g), torch.stack(y)

    def setup(self, stage: Optional[str] = None):
        pass

    def prepare_data(self, *args, **kwargs):
        pass
