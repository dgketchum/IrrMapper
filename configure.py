import os
import json
from pathlib import Path
from argparse import Namespace

import torch

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

    gpu_map = {'V100': 3,
               'RTX': 1,
               'K40': 1.5}

    b_sizes = {'pixel': 3,
               'image': 12,
               'temporal_image': 6}

    params.batch_size = int(b_sizes[params.mode] * gpu_map[params.gpu])

    data = '/media/nvm/ts_data'
    if not os.path.isdir(data):
        data = '/nobackup/dketchu1/ts_data'
    pixels = os.path.join(data, 'pixels')
    images = os.path.join(data, 'images')

    device_ct = torch.cuda.device_count()
    print('device count: {}'.format(device_ct))

    config = {'model': params.model,
              'mode': params.mode,
              'input_dim': 6,
              'dataset_folder': data,
              'rdm_seed': 1,
              'epochs': 100,
              'lr': 0.0001,
              'n_classes': N_CLASSES,
              'device_ct': device_ct,
              'node_ct': params.nodes,
              'num_workers': params.workers,
              'machine': params.machine,
              'batch_size': params.batch_size,
              'sample_n': [0.0, 0.49362077, 0.45154002, 0.05483921],
              'res_dir': os.path.join(path[0], 'models', params.model, 'results'),
              }

    if config['model'] == 'dcm':
        config['dataset_folder'] = pixels
        config['predict_mode'] = 'pixel'
        config['num_classes'] = 3
        config['hidden_size'] = 18
        config['num_layers'] = 2
        config['bidirectional'] = True
        config['seed'] = 121
        config['lr'] = 0.0025
        config['res_dir'] = os.path.join(path[0], 'models', 'dcm', 'results')
        with open(os.path.join(path[0], 'models', 'dcm', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'nnet':
        config['dataset_folder'] = pixels
        config['predict_mode'] = 'pixel'
        config['hidden_size'] = 256
        config['seed'] = 121
        config['lr'] = 0.0025
        config['res_dir'] = os.path.join(path[0], 'models', 'nnet', 'results')
        with open(os.path.join(path[0], 'models', 'nnet', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'tcnn':
        config['dataset_folder'] = pixels
        config['predict_mode'] = 'pixel'
        config['sequence_len'] = SEQUENCE_LEN
        config['lr'] = 0.0025
        config['hidden_dim'] = 4
        config['nker'] = '[16, 16, 16]'
        config['mlp3'] = '[16, 4]'
        config['res_dir'] = os.path.join(path[0], 'models', 'temp_cnn', 'results')
        with open(os.path.join(path[0], 'models', 'temp_cnn', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'clstm':
        config['dataset_folder'] = images
        config['predict_mode'] = 'temporal_image'
        config['input_dim'] = 7
        config['num_layers'] = 1
        config['kernel_size'] = (3, 3)
        config['hidden_dim'] = 4
        config['res_dir'] = os.path.join(path[0], 'models', 'conv_lstm', 'results')
        with open(os.path.join(path[0], 'models', 'conv_lstm', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'unet':
        config['dataset_folder'] = images
        config['predict_mode'] = 'image'
        config['seed'] = 121
        config['lr'] = 0.0001

    return Namespace(**config)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
