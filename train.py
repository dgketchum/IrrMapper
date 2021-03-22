import os
import json
from datetime import datetime
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.unet.unet import UNet
from configure import get_config


def prepare_output(config):
    dt = datetime.now().strftime('{}-%Y.%m.%d.%H.%M-{}-{}'.format(config.machine,
                                                                  config.model,
                                                                  config.mode))
    new_dir = os.path.join(config.res_dir, dt)
    os.makedirs(new_dir, exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'checkpoints'), exist_ok=True)
    with open(os.path.join(new_dir, 'config.json'), 'w') as file:
        file.write(json.dumps(vars(config), indent=4))
    return new_dir


def main(params):

    config = get_config(**vars(params))

    model = UNet(config)

    log_dir = prepare_output(config)
    logger = TensorBoardLogger(log_dir, name='log')

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        save_top_k=1,
        monitor='val_acc',
        verbose=True)

    stop_callback = EarlyStopping(
        monitor='val_acc',
        mode='auto',
        patience=15,
        verbose=False)

    trainer = Trainer(
        precision=16,
        min_epochs=100,
        limit_val_batches=250,
        # overfit_batches=100,
        auto_lr_find=True,
        gpus=config.device_ct,
        num_nodes=config.node_ct,
        callbacks=[checkpoint_callback, stop_callback],
        progress_bar_refresh_rate=params.progress,
        log_every_n_steps=5,
        logger=logger)

    trainer.tune(model)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model', default='unet')
    parser.add_argument('--mode', default='image')
    parser.add_argument('--gpu', default='RTX')
    parser.add_argument('--machine', default='pc')
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--progress', default=0, type=int)
    parser.add_argument('--workers', default=6, type=int)
    args = parser.parse_args()
    main(args)
# ========================================================================================
