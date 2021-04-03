from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.standard import StandardModule
from data_load.dataset import IrrMapDataModule


class NNet(StandardModule):
    def __init__(self, **hparams):
        StandardModule.__init__(self, **hparams)

        self.classifier = nn.Sequential(
            nn.Linear(self.n_channels, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
            nn.BatchNorm1d(int(self.hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size / 2), self.n_classes))

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_f1 = pl.metrics.FBeta(num_classes=self.n_classes, beta=1)
        self.valid_rec = pl.metrics.Precision(num_classes=self.n_classes, mdmc_average='global')
        self.valid_prec = pl.metrics.Recall(num_classes=self.n_classes, mdmc_average='global')

    def forward(self, x):
        x = x.squeeze()
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict_example(self, x, g, y):
        out = self.forward(x)
        pred = out.argmax(1)
        x = x.reshape(x.shape[0], g.shape[-1], g.shape[-1], x.shape[-1])
        x = x.squeeze().permute(2, 0, 1).numpy()
        pred = pred.reshape(y.shape).squeeze().numpy()
        g = g.squeeze().numpy()
        y = y.squeeze().numpy()
        return x, g, y, pred

    @staticmethod
    def validation_end(outputs):
        avg_loss = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_acc': avg_loss}
        return {'avg_val_acc': avg_loss, 'log': tensorboard_logs}

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

    def configure_model(self):
        for name, val in self.hparams.items():
            setattr(self, name, val)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
