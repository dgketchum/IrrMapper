""""Credit: https://github.com/locuslab/TCN"""
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pytorch_lightning as pl

from models.standard import StandardModule
from data_load.dataset import IrrMapDataModule
from configure import BANDS, CHANNELS, TERRAIN, SEQUENCE_LEN
from data_load import NORM


# TODO: batch norm

class TempConv(StandardModule):
    def __init__(self, **hparams):
        StandardModule.__init__(self, **hparams)
        hid_channels = [25 for x in range(0, 8)]
        self.tcn = TemporalConvNet(self.n_channels, hid_channels, kernel_size=self.kernel_size)
        self.linear = nn.Linear(hid_channels[-1], self.n_classes)
        self.init_weights()

        self.mean = torch.tensor(NORM[0][:91]).float()
        self.std = torch.tensor(NORM[1][:91]).float()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])

    def predict_example(self, x, g, y):
        def inv_transform_(x):
            print(x.size(), self.std.size())
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            x = x.mul_(self.std).add_(self.mean)
            return x

        x = x.squeeze()
        out = self.forward(x)
        pred = out.argmax(1)
        pred = pred.reshape(g.shape[-1], g.shape[-1]).squeeze().numpy()
        g = g.squeeze().numpy()
        y = y.squeeze().numpy()
        x = inv_transform_(x)
        x = x.reshape(x.shape[-1], g.shape[-1], g.shape[-1])
        return x, g, y, pred

    def __dataloader(self):
        itdl = IrrMapDataModule(self.hparams)
        loaders = {'train': itdl.train_dataloader(),
                   'val': itdl.val_loader()}
        dct = deepcopy(self.hparams)
        dct['dataset_folder'] = dct['dataset_folder'].replace('pixels', 'images')
        dct['mode'] = 'image'
        loaders['test'] = IrrMapDataModule(dct).test_loader()
        return loaders

    def train_dataloader(self):
        return self.__dataloader()['train']

    def val_dataloader(self):
        return self.__dataloader()['val']

    def test_dataloader(self):
        return self.__dataloader()['test']


class Chomp1d(pl.LightningModule):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(pl.LightningModule):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        # self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        # self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(pl.LightningModule):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            pad = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=pad, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
