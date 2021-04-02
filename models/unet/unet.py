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

from data_load.dataset import IrrMapDataModule


class UNet(pl.LightningModule):
    def __init__(self, hparams, bilinear=True):
        super(UNet, self).__init__()

        self.hparams = hparams
        self.configure_model()

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='max',
                                                               factor=0.5,
                                                               patience=5,
                                                               verbose=True)

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'interval': 'epoch',
                                 'monitor': 'val_acc'}}

    def cross_entropy_loss(self, logits, labels):
        weights = torch.tensor(self.hparams['sample_n'], dtype=torch.float32, device=self.device)
        loss = nn.CrossEntropyLoss(ignore_index=0, weight=weights)
        return loss(logits, labels)

    def _mask_out(self, y, logits):
        mask = y.flatten() > 0
        y = y.flatten()[mask]
        pred = torch.softmax(logits, 1)
        pred = torch.argmax(pred, dim=1).flatten()[mask]
        return y, pred

    def training_step(self, batch, batch_idx):
        x, g, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        y, pred = self._mask_out(y, logits)
        self.log('train_acc_step', self.train_acc(pred, y), prog_bar=True, logger=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.log('train_acc_epoch', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x, g, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        y, pred = self._mask_out(y, logits)
        acc = self.valid_acc(pred, y)
        f1 = self.valid_f1(pred, y)
        rec = self.valid_rec(pred, y)
        prec = self.valid_prec(pred, y)
        self.log('val_acc', acc, on_epoch=True)
        self.log('val_f1', f1, on_epoch=True)
        self.log('val_rec', rec, on_epoch=True)
        self.log('val_prec', prec, on_epoch=True)
        return {'val_loss': loss,
                'val_acc': acc,
                'val_f1': f1,
                'val_rec': rec,
                'val_prec': prec}

    def test_step(self, batch, batch_idx):
        x, g, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('test_loss', loss)
        y, pred = self._mask_out(y, logits)
        self.log('test_acc', self.valid_acc(pred, y), on_epoch=True)
        self.log('test_f1', self.valid_f1(pred, y), on_epoch=True)
        self.log('test_rec', self.valid_rec(pred, y), on_epoch=True)
        self.log('test_prec', self.valid_prec(pred, y), on_epoch=True)
        return {'test_acc': self.valid_acc(pred, y)}

    @staticmethod
    def validation_end(outputs):
        avg_loss = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_acc': avg_loss}
        return {'avg_val_acc': avg_loss, 'log': tensorboard_logs}

    def __dataloader(self):
        itdl = IrrMapDataModule(self.hparams)
        loaders = {'train': itdl.train_dataloader(),
                   'val': itdl.val_loader(),
                   'test': itdl.test_loader()}
        return loaders

    def train_dataloader(self):
        return self.__dataloader()['train']

    def val_dataloader(self):
        return self.__dataloader()['val']

    def test_dataloader(self):
        return self.__dataloader()['test']

    def configure_model(self):
        for name, val in self.hparams.items():
            setattr(self, name, val)


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
