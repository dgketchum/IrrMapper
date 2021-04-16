import os
from pathlib import Path
from argparse import Namespace

import torch
from pytorch_lightning.utilities import seed

seed.seed_everything(42)

path = Path(__file__).parents
N_CLASSES = 4

CHANNELS = 7
BANDS = 91
TERRAIN = 5
CDL = 2
SEQUENCE_LEN = 13

PIXEL_CLASS_DIST = [6618464, 7235264, 59574370]


# CHANNELS = 7
# BANDS = 70
# TERRAIN = 5
# SEQUENCE_LEN = 10


def get_config(**params):
    params = Namespace(**params)

    # local
    data = '/media/nvm/ts_data/{}'.format(params.stack)
    # nasa nas
    if not os.path.isdir(data):
        data = '/nobackup/dketchu1/ts_data/{}'.format(params.stack)
    # aws
    if not os.path.isdir(data):
        data = '/home/ubuntu/data/ts_data/{}'.format(params.stack)

    pixels = os.path.join(data, 'pixels')
    images = os.path.join(data, 'images')

    device_ct = torch.cuda.device_count()

    config = {'model': params.model,
              'stack': params.stack,
              'dataset_folder': data,
              'epochs': 100,
              'n_classes': N_CLASSES,
              'device_ct': device_ct,
              'node_ct': params.nodes,
              'num_workers': params.workers,
              'machine': params.machine,
              'sample_n': [0.0, 0.49362077, 0.45154002, 0.05483921],
              'res_dir': os.path.join(path[0], 'models', params.model, 'results'),
              }

    if config['model'] == 'dcm':
        config['dataset_folder'] = pixels
        config['mode'] = 'pixel'
        config['num_classes'] = 3
        config['hidden_size'] = 18
        config['num_layers'] = 2
        config['bidirectional'] = True
        config['lr'] = 0.0025

    if config['model'] == 'nnet':
        config['dataset_folder'] = pixels
        config['n_channels'] = BANDS
        config['mode'] = 'pixel'
        config['hidden_size'] = 256
        config['lr'] = 0.001

    if config['model'] == 'tcnn':
        config['dataset_folder'] = pixels
        config['input_dim'] = 7
        config['mode'] = 'pixel'
        config['sequence_len'] = SEQUENCE_LEN
        config['n_channels'] = CHANNELS
        config['lr'] = 0.001
        config['hidden_dim'] = 4
        config['kernel_size'] = 3

    if config['model'] == 'clstm':
        config['dataset_folder'] = images
        config['mode'] = 'temporal_image'
        config['input_dim'] = 7
        config['num_layers'] = 1
        config['kernel_size'] = (3, 3)
        config['hidden_dim'] = 4

    if config['model'] == 'unet':
        config['dataset_folder'] = images
        config['unet_dim_seed'] = 32
        config['mode'] = 'image'
        config['n_channels'] = BANDS
        config['lr'] = 0.001

    gpu_map = {'V100': 3,
               'RTX': 1,
               'K40': 1.5}

    b_sizes = {'pixel': 1,
               'image': 12,
               'temporal_image': 6}

    config['batch_size'] = int(b_sizes[config['mode']] * gpu_map[params.gpu])

    return Namespace(**config)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
