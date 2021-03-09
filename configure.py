import os
import json
from pathlib import Path

import torch

path = Path(__file__).parents

CHANNELS = 7
BANDS = 91
TERRAIN = 5
SEQUENCE_LEN = 13


# CHANNELS = 7
# BANDS = 70
# TERRAIN = 5
# SEQUENCE_LEN = 10


def get_config(model='clstm', mode='irr'):
    data = '/media/hdisk/t_data/tarchives'
    pixels = os.path.join(data, 'pixels')
    images = os.path.join(data, 'images')

    if not os.path.isdir(data):
        data = '/mnt/beegfs/dk128872/ts_data/cmask/tar'
        pixels = os.path.join(data, 'pixels')
        images = os.path.join(data, 'images')
        pixel_sets = os.path.join(data, 'pixel_sets')

    device_ct = torch.cuda.device_count()

    config = {'model': model,
              'mode': mode,
              'rdm_seed': 1,
              'display_step': 100,
              'epochs': 100,
              'num_classes': 4,
              'input_dim': CHANNELS,
              'geomfeat': None,
              'device': 'cuda:0',
              'num_workers': 4,
              'pooling': 'mean_std',
              'dropout': 0.2,
              'gamma': 1,
              'alpha': None,
              'prediction_dir': os.path.join(images, 'test'),
              'norm': os.path.join(data, 'pixels', 'meanstd.pkl'), }

    if config['model'] == 'dcm':
        config['dataset_folder'] = pixels
        config['predict_mode'] = 'pixel'
        config['batch_size'] = 1 * device_ct
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
        config['batch_size'] = 1 * device_ct
        config['hidden_size'] = 256
        config['input_dim'] = BANDS
        config['seed'] = 121
        config['lr'] = 0.0025
        config['res_dir'] = os.path.join(path[0], 'models', 'nnet', 'results')
        with open(os.path.join(path[0], 'models', 'nnet', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'tcnn':
        config['dataset_folder'] = pixels
        config['predict_mode'] = 'pixel'
        config['batch_size'] = 1 * device_ct
        config['sequence_len'] = SEQUENCE_LEN
        config['hidden_dim'] = 4
        config['nker'] = '[16, 16, 16]'
        config['mlp3'] = '[16, 4]'
        config['res_dir'] = os.path.join(path[0], 'models', 'temp_cnn', 'results')
        with open(os.path.join(path[0], 'models', 'temp_cnn', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'clstm':
        config['dataset_folder'] = images
        config['predict_mode'] = 'temporal_image'
        config['batch_size'] = 16 * device_ct
        config['input_dim'] = 7
        config['num_layers'] = 1
        config['kernel_size'] = (3, 3)
        config['hidden_dim'] = 4
        config['res_dir'] = os.path.join(path[0], 'models', 'conv_lstm', 'results')
        with open(os.path.join(path[0], 'models', 'conv_lstm', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'unet':
        config['dataset_folder'] = images
        config['batch_size'] = 4 * device_ct
        config['input_dim'] = BANDS
        config['predict_mode'] = 'image'
        # config['sample_n'] = [1012505459, 337955249, 43964057, 153976261, 222513580, 69336274]
        config['seed'] = 121
        config['lr'] = 0.0001
        config['res_dir'] = os.path.join(path[0], 'models', config['model'], 'results')
        with open(os.path.join(path[0], 'models', config['model'], 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    return config


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
