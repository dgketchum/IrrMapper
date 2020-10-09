# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
import os
import json
from pathlib import Path

path = Path(__file__).parents


def get_config():
    data = '/home/dgketchum/IrrigationGIS/tfrecords/tarchives'

    if not os.path.isdir(data):
        data = '/mnt/beegfs/dk128872/ts_data/cmask/tar'

    config = {'mode': 'irr',
              'rdm_seed': 1,
              'epochs': 100,
              'display_step': 10,
              'num_classes': 4,
              'nomenclature': 'label_4class',
              'kfold': 5,
              'input_dim': 7,
              'geomfeat': None,
              'device': 'cuda:0',
              'num_workers': 1,
              'pooling': 'mean_std',
              'dropout': 0.2,
              'gamma': 1,
              'alpha': None,
              'prediction_dir': os.path.join(data, 'images', 'test'),
              'model': 'tcnn'}

    if config['model'] == 'ltae':
        config['dataset_folder'] = os.path.join(data, 'pixel_sets')
        config['batch_size'] = 128
        config['mlp1'] = '[7, 32, 64]'
        config['mlp2'] = '[131, 128]'
        config['mlp3'] = '[256,128]'
        config['mlp4'] = '[128, 64, 32, 20]'
        config['n_head'] = 16
        config['d_k'] = 8
        config['d_model'] = 256
        config['T'] = 1000
        config['positions'] = None
        config['geom_dim'] = 3
        config['geomfeat'] = True
        config['lms'] = 13
        config['npixel'] = 64
        config['subset'] = None
        config['lr'] = 0.00025
        config['num_classes'] = 4
        config['res_dir'] = os.path.join(path[0], 'models', 'ltae_pse', 'results')
        with open(os.path.join(path[0], 'models', 'ltae_pse', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))
            # exit()

    if config['model'] == 'dcm':
        config['batch_size'] = 1
        config['dataset_folder'] = os.path.join(data, 'pixels')
        config['hidden_size'] = 256
        config['num_layers'] = 2
        config['bidirectional'] = True
        config['seed'] = 121
        config['lr'] = 0.00025
        config['res_dir'] = os.path.join(path[0], 'models', 'dcm', 'results')
        with open(os.path.join(path[0], 'models', 'dcm', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'tcnn':
        config['batch_size'] = 1
        config['dataset_folder'] = os.path.join(data, 'pixels')
        config['sequence_len'] = 13
        config['nker'] = '[16, 16, 16]'
        config['mlp3'] = '[16, 16]'
        config['res_dir'] = os.path.join(path[0], 'models', 'temp_cnn', 'results')
        with open(os.path.join(path[0], 'models', 'temp_cnn', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['model'] == 'clstm':
        config['batch_size'] = 8
        config['input_dim'] = 7
        config['num_layers'] = 1
        config['dataset_folder'] = os.path.join(data, 'images')
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
