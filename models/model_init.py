import pickle as pkl

import torch.utils.data as data

from data_prep.pse_dataset import PixelSetData
from data_prep.pixel_dataset import pixel_dataset
from data_prep.image_dataset import image_dataset

from models.ltae_pse.stclassifier import PseLTae
from models.dcm.dcm import DCM
from models.temp_cnn.temp_cnn import TempConv
from models.conv_lstm.conv_lstm import ConvLSTM






def get_predict_loader(config):
    mean_std = pkl.load(open(config['image_norm'], 'rb'))
    dt = image_dataset('test', config['prediction_dir'], mean_std)
    test = get_dataloader(dt, config)
    return test


def get_loaders(config):
    dt = None
    splits = ['train', 'test', 'valid']
    extra = 'geomfeat' if config['geomfeat'] else None

    if config['model'] in ['tcnn', 'dcm']:
        mean_std = pkl.load(open(config['pixel_norm'], 'rb'))
        dt = (pixel_dataset(split, config, mean_std) for split in splits)

    if config['model'] == 'ltae':
        mean_std = pkl.load(open(config['pixel_norm'], 'rb'))
        dt = PixelSetData(config['dataset_folder'], labels=config['nomenclature'],
                          npixel=config['npixel'],
                          sub_classes=config['subset'],
                          norm=mean_std,
                          extra_feature=extra)

    if config['model'] == 'clstm':
        mean_std = pkl.load(open(config['image_norm'], 'rb'))
        dt = (image_dataset(split, config, mean_std) for split in splits)

    train, test, valid = (get_dataloader(d, config) for d in dt)
    return train, test, valid


def get_model(config):
    model = None
    if config['model'] == 'dcm':
        model_config = dict(input_dim=config['input_dim'], hidden_size=config['hidden_size'], seed=config['seed'],
                            num_layers=config['num_layers'], bidirectional=config['bidirectional'],
                            dropout=config['dropout'], num_classes=config['num_classes'])
        model = DCM(**model_config)

    elif config['model'] == 'tcnn':
        model_config = dict(input_dim=config['input_dim'], nker=config['nker'], seq_len=config['sequence_len'],
                            nfc=config['mlp3'])
        model = TempConv(**model_config)

    elif config['model'] == 'ltae':
        model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                            mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                            dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                            mlp4=config['mlp4'], d_model=config['d_model'], return_att=True)
        if config['geomfeat']:
            model_config.update(with_extra=True, extra_size=config['geom_dim'])
        else:
            model_config.update(with_extra=False, extra_size=None)
        model = PseLTae(**model_config)

    elif config['model'] == 'clstm':
        model_config = dict(input_dim=config['input_dim'], kernel_size=config['kernel_size'],
                            hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
                            batch_first=True, bias=True, return_all_layers=False)
        model = ConvLSTM(**model_config)

    return model


def get_dataloader(dt, config):
    loader = data.DataLoader(dt, batch_size=config['batch_size'],
                             num_workers=config['num_workers'], pin_memory=False)
    return loader


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
