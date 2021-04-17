import os
import json
from datetime import datetime
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from models.unet.unet import UNet
from models.nnet.nnet import NNet
from models.tcnn.tcnn import TempConv
from configure import get_config


MODEL_MAP = {'unet': UNet,
             'nnet': NNet,
             'tcnn': TempConv}


def prepare_output(config):
    dt = datetime.now().strftime('{}-%Y.%m.%d.%H.%M-{}-{}'.format(config.machine,
                                                                  config.model,
                                                                  config.mode))
    new_dir = os.path.join(config.res_dir, dt)
    os.makedirs(new_dir, exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'figures'), exist_ok=True)
    with open(os.path.join(new_dir, 'config.json'), 'w') as file:
        file.write(json.dumps(vars(config), indent=4))
    return new_dir


def main(params):

    config = get_config(**vars(params))

    model = MODEL_MAP[config.model]
    model = model(**vars(config))

    log_dir = prepare_output(config)
    logger = TensorBoardLogger(log_dir, name='log')

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        verbose=True)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    accelerator = 'ddp' if config.device_ct > 1 else None
    ddp_plug = DDPPlugin(find_unused_parameters=False) if config.device_ct > 1 else None

    trainer = Trainer(
        precision=16,
        min_epochs=150,
        accelerator=accelerator,
        plugins=ddp_plug,
        gpus=config.device_ct,
        num_nodes=config.node_ct,
        callbacks=[checkpoint_callback, lr_monitor],
        progress_bar_refresh_rate=params.progress,
        log_every_n_steps=5,
        logger=logger)

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model', default='unet')
    parser.add_argument('--gpu', default='RTX')
    parser.add_argument('--machine', default='pc')
    parser.add_argument('--stack', default='nm')
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--progress', default=0, type=int)
    parser.add_argument('--workers', default=16, type=int)
    args = parser.parse_args()
    main(args)
# ========================================================================================
