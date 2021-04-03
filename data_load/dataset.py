import os
from glob import glob
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from configure import BANDS, CHANNELS, TERRAIN, SEQUENCE_LEN

SELECT_CHANNELS = torch.tensor([44, 45, 48, 65, 66, 69])

"""WARNING This dataloader hijacks 'test' data and puts it in the training data stack, when test_loader is called
    it is just getting the valid data. For pixel models, it pulls valid data in image format, rather than pixel stack"""


class IrrMapDataset(Dataset):

    def __init__(self, data_dir, mode, transforms=None, model=None):
        self.data_dir = data_dir
        self.mode = mode
        self.model = model
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
            x = x.reshape((x.shape[0], SEQUENCE_LEN, CHANNELS))
            g = g.float()
            y = img[:, -1].long()
            # y -= 1

            if self.model == 'nnet':
                x = x.reshape((x.shape[0], x.shape[1] * x.shape[2])).float()

            if self.model == 'tcnn':
                x = x.permute(0, 2, 1).float()

        elif self.mode == 'image':
            features = img[:, :, :BANDS + TERRAIN]
            features = features.float()
            x = features[:, :, :BANDS]
            x = x.permute((2, 0, 1)).float()
            # x = x.index_select(dim=0, index=SELECT_CHANNELS)
            y = img[:, :, -1].long()
            g = features[:, :, BANDS:BANDS + TERRAIN].permute(2, 0, 1)
            # give C x H x W

            if self.model == 'nnet':
                x = x.reshape((x.shape[0], x.shape[1] * x.shape[2])).permute(1, 0).float()

            if self.model == 'tcnn':
                x = x.permute(0, 2, 1).float()

        elif self.mode == 'temporal_image':
            features = img[:, :, :BANDS + TERRAIN]
            x = features[:, :, :BANDS]
            x = x.reshape(x.shape[0], x.shape[1], SEQUENCE_LEN, CHANNELS)
            x = x.permute((2, 3, 0, 1)).float()
            y = img[:, :, -4:].permute(2, 0, 1).int()
            g = features[:, :, BANDS:BANDS + TERRAIN].permute(2, 0, 1)

        else:
            raise KeyError(
                'Must choose from {} image modes'.format(['pixel',
                                                          'image',
                                                          'temporal_image']))

        if not torch.isfinite(img).all():
            print('non-finite in {}'.format(img_path))
            if not torch.isfinite(x).all():
                raise ArithmeticError('lalbel has nan/inf')
            if not torch.isfinite(y).all():
                raise ArithmeticError('lalbel has nan/inf')

        if self.transforms:
            x = self.transforms(features)

        return x, g, y

    def __len__(self):
        return len(self.img_paths)


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
