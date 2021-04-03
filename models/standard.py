from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl

from data_load.dataset import IrrMapDataModule

"""Standard Module holds all methods we want standard accross models"""


class StandardModule(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()

        self.hparams = hparams
        self.configure_model()

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_f1 = pl.metrics.FBeta(num_classes=self.n_classes, beta=1)
        self.valid_rec = pl.metrics.Precision(num_classes=self.n_classes, mdmc_average='global')
        self.valid_prec = pl.metrics.Recall(num_classes=self.n_classes, mdmc_average='global')

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
        x, g, y = torch.squeeze(x), torch.squeeze(g), torch.squeeze(y)
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
        x, g, y = torch.squeeze(x), torch.squeeze(g), torch.squeeze(y)
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
        x, g, y = torch.squeeze(x), torch.squeeze(g), torch.squeeze(y)
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('test_loss', loss)
        y, pred = self._mask_out(y, logits)
        self.log('test_acc', self.valid_acc(pred, y), on_epoch=True)
        self.log('test_f1', self.valid_f1(pred, y), on_epoch=True)
        self.log('test_rec', self.valid_rec(pred, y), on_epoch=True)
        self.log('test_prec', self.valid_prec(pred, y), on_epoch=True)
        return {'test_acc': self.valid_acc(pred, y)}

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
                   'valid': itdl.val_loader(),
                   'test': itdl.test_loader()}
        return loaders

    def train_dataloader(self):
        return self.__dataloader()['train']

    def val_dataloader(self):
        return self.__dataloader()['valid']

    def test_dataloader(self):
        return self.__dataloader()['test']

    def configure_model(self):
        for name, val in self.hparams.items():
            setattr(self, name, val)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
