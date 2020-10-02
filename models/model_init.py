import numpy as np
import pickle as pkl

import torch.utils.data as data

from data_prep.pse_dataset import PixelSetData
from data_prep.pixel_dataset import pixel_data
from data_prep.image_dataset import ImageDataset, image_dataset

from models.ltae_pse.stclassifier import PseLTae
from models.dcm.dcm import DCM
from models.temp_cnn.temp_cnn import TempConv
from models.conv_lstm.conv_lstm import ConvLSTM


def get_loaders(config):
    train, test, valid = None, None, None
    train_loc, test_loc = '{}/train'.format(config['dataset_folder']), \
                          '{}/test'.format(config['dataset_folder'])
    valid_loc = config['validation_folder']
    data_locations = [train_loc, test_loc, valid_loc]

    mean_std = pkl.load(open(config['dataset_folder'] + '/meanstd.pkl', 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None

    if config['dcm']:
        dt = pixel_data(config['dataset_folder'], labels=config['nomenclature'], norm=mean_std, extra_feature=None)
        train, test = get_dataloader(dt, config)

    if config['tcnn']:
        dt = pixel_data(config['dataset_folder'], labels=config['nomenclature'], norm=mean_std,
                        extra_feature=None)
        train, test = get_dataloader(dt, config)

    if config['ltae']:
        dt = PixelSetData(config['dataset_folder'], labels=config['nomenclature'], npixel=config['npixel'],
                          sub_classes=config['subset'],
                          norm=mean_std,
                          extra_feature=extra)
        train, test = get_dataloader(dt, config)

    if config['clstm']:
        dt = (image_dataset(loc) for loc in ['train', 'test', 'valid'])
        train, test, valid = (get_dataloader(d, config) for d in dt)

    return train, test, valid


def get_model(config):
    model = None
    if config['dcm']:
        model_config = dict(input_dim=config['input_dim'], hidden_size=config['hidden_size'], seed=config['seed'],
                            num_layers=config['num_layers'], bidirectional=config['bidirectional'],
                            dropout=config['dropout'], num_classes=config['num_classes'])
        model = DCM(**model_config)

    elif config['tcnn']:
        model_config = dict(input_dim=config['input_dim'], nker=config['nker'], seq_len=config['sequence_len'],
                            nfc=config['mlp3'])
        model = TempConv(**model_config)

    elif config['ltae']:
        model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                            mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                            dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                            mlp4=config['mlp4'], d_model=config['d_model'], return_att=True)
        if config['geomfeat']:
            model_config.update(with_extra=True, extra_size=config['geom_dim'])
        else:
            model_config.update(with_extra=False, extra_size=None)
        model = PseLTae(**model_config)

    elif config['clstm']:
        model_config = dict(input_dim=config['input_dim'], kernel_size=config['kernel_size'],
                            hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
                            batch_first=True, bias=True, return_all_layers=False)
        model = ConvLSTM(**model_config)

    return model


def get_dataloader(dt, config):
    loader = data.DataLoader(dt, batch_size=config['batch_size'],
                             num_workers=config['num_workers'], pin_memory=True)
    return loader


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
