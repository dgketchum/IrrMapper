import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pytorch_lightning as pl

from data_load.dataset import IrrMapDataModule


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
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
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
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TempConv(pl.LightningModule):
    def __init__(self, hparams):
        super(TempConv, self).__init__()
        self.hparams = hparams
        self.configure_model()
        self.tcn = TemporalConvNet(self.input_dim, self.n_channels, kernel_size=self.kernel_size)
        self.linear = nn.Linear(self.n_channels[-1], self.output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1]), None

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
        self.log('val_acc', self.valid_acc(pred, y), on_epoch=True)
        self.log('val_f1', self.valid_f1(pred, y), on_epoch=True)
        self.log('val_rec', self.valid_rec(pred, y), on_epoch=True)
        self.log('val_prec', self.valid_prec(pred, y), on_epoch=True)
        return {'val_acc': self.valid_acc(pred, y)}

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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
