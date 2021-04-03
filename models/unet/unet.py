"""
Credits to:
code author: Milese Alexandre
code: https://github.com/milesial/Pytorch-UNet
paper: Ronneberger, O., Fischer, P. and Brox, T., 2015, October.
U-net: Convolutional networks for biomedical image segmentation.
In International Conference on Medical image computing
and computer-assisted intervention (pp. 234-241).
Springer, Cham.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from models.standard import StandardModule


class UNet(StandardModule):
    def __init__(self, hparams, bilinear=True):
        StandardModule.__init__(self, hparams)

        seed = 64
        self.inc = DoubleConv(self.n_channels, seed)
        self.down1 = Down(seed, seed * 2)
        self.down2 = Down(seed * 2, seed * 4)
        self.down3 = Down(seed * 4, seed * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(seed * 8, seed * 16 // factor)
        self.up1 = Up(seed * 16, seed * 8 // factor, bilinear)
        self.up2 = Up(seed * 8, seed * 4 // factor, bilinear)
        self.up3 = Up(seed * 4, seed * 2 // factor, bilinear)
        self.up4 = Up(seed * 2, seed, bilinear)
        self.outc = OutConv(seed, self.n_classes)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_f1 = pl.metrics.FBeta(num_classes=self.n_classes, beta=1)
        self.valid_rec = pl.metrics.Precision(num_classes=self.n_classes, mdmc_average='global')
        self.valid_prec = pl.metrics.Recall(num_classes=self.n_classes, mdmc_average='global')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def predict_example(self, x, g, y):
        out = self.forward(x)
        pred = out.argmax(1)
        x, g = x.squeeze().numpy(), g.squeeze().numpy()
        y, pred = y.squeeze().numpy(), pred.squeeze().numpy()
        return x, g, y, pred

    @staticmethod
    def validation_end(outputs):
        avg_loss = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_acc': avg_loss}
        return {'avg_val_acc': avg_loss, 'log': tensorboard_logs}



class DoubleConv(pl.LightningModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(pl.LightningModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(pl.LightningModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
