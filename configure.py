import os
import json
from pathlib import Path

import torch

path = Path(__file__).parents


def get_config(model='clstm', mode='irr'):
    data = '/home/dgketchum/tfrecords/tarchives'
    pixels = os.path.join(data, 'pixels')
    images = os.path.join(data, 'images')
    pixel_sets = os.path.join(data, 'pixel_sets')

    if not os.path.isdir(data):
        data = '/mnt/beegfs/dk128872/ts_data/cmask/tar'
        pixels = os.path.join(data, 'pixels')
        images = os.path.join(data, 'images')
        pixel_sets = os.path.join(data, 'pixel_sets')

    device_ct = torch.cuda.device_count()

    config = {'model': model,
              'mode': mode,
              'rdm_seed': 1,
              'display_step': 50,
              'epochs': 1,
              'num_classes': 4,
              'input_dim': 7,
              'geomfeat': None,
              'device': 'cuda:0',
              'num_workers': 1,
              'pooling': 'mean_std',
              'dropout': 0.2,
              'gamma': 1,
              'alpha': None,
              'prediction_dir': os.path.join(images, 'test'),
              'norm': os.path.join(data, 'images', 'meanstd.pkl'), }

    if config['model'] == 'ltae':
        config['dataset_folder'] = pixel_sets
        config['batch_size'] = 128 * device_ct
        config['mlp1'] = '[7, 32, 64]'
        config['mlp2'] = '[128, 128]'
        config['mlp3'] = '[256,128]'
        config['mlp4'] = '[128, 64, 32, 4]'
        config['n_head'] = 16
        config['d_k'] = 8
        config['d_model'] = 256
        config['T'] = 1000
        config['positions'] = None
        config['geom_dim'] = 5
        config['geomfeat'] = False
        config['lms'] = 13
        config['n_pixel'] = 64
        config['subset'] = None
        config['lr'] = 0.00025
        config['num_classes'] = 4
        config['res_dir'] = os.path.join(path[0], 'models', 'ltae_pse', 'results')
        with open(os.path.join(path[0], 'models', 'ltae_pse', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'dcm':
        config['dataset_folder'] = pixels
        config['predict_mode'] = 'pixel'
        config['batch_size'] = 1 * device_ct
        config['hidden_size'] = 36
        config['num_layers'] = 2
        config['bidirectional'] = True
        config['seed'] = 121
        config['lr'] = 0.00025
        config['res_dir'] = os.path.join(path[0], 'models', 'dcm', 'results')
        with open(os.path.join(path[0], 'models', 'dcm', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'tcnn':
        config['dataset_folder'] = pixels
        config['predict_mode'] = 'pixel'
        config['batch_size'] = 1 * device_ct
        config['sequence_len'] = 13
        config['nker'] = '[16, 16, 16]'
        config['mlp3'] = '[16, 4]'
        config['res_dir'] = os.path.join(path[0], 'models', 'temp_cnn', 'results')
        with open(os.path.join(path[0], 'models', 'temp_cnn', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'clstm':
        config['dataset_folder'] = images
        config['predict_mode'] = 'image'
        config['batch_size'] = 16 * device_ct
        config['input_dim'] = 7
        config['num_layers'] = 1
        config['kernel_size'] = (3, 3)
        config['hidden_dim'] = 4
        config['res_dir'] = os.path.join(path[0], 'models', 'conv_lstm', 'results')
        with open(os.path.join(path[0], 'models', 'conv_lstm', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    for k, v in config.items():
        if 'mlp' in k or k == 'nker':
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))

    return config


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
