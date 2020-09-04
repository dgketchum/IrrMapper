import numpy as np
import pickle as pkl

import torch.utils.data as data
from sklearn.model_selection import KFold

from data_prep.pse_dataset import PixelSetData
from data_prep.pixel_dataset import pixel_data
from data_prep.image_dataset import ImageDataset

from models.ltae_pse.stclassifier import PseLTae
from models.dcm.dcm import DCM
from models.temp_cnn.temp_cnn import TempConv
from models.conv_lstm.conv_lstm import ConvLSTM


def get_loaders(config):
    mean_std = pkl.load(open(config['dataset_folder'] + '/meanstd.pkl', 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None
    if config['dcm']:
        dt = pixel_data(config['dataset_folder'], labels=config['nomenclature'], norm=mean_std, extra_feature=None)
        loaders = _dataloader_split(dt, config['kfold'], config)

    if config['tcnn']:
        dt = pixel_data(config['dataset_folder'], labels=config['nomenclature'], norm=mean_std,
                        extra_feature=None)
        loaders = _dataloader_split(dt, config['kfold'], config)

    if config['ltae']:
        dt = PixelSetData(config['dataset_folder'], labels=config['nomenclature'], npixel=config['npixel'],
                          sub_classes=config['subset'],
                          norm=mean_std,
                          extra_feature=extra)
        loaders = _dataloader_split(dt, config['kfold'], config)

    if config['clstm']:
        dt = ImageDataset(config['dataset_folder'], norm=mean_std)
        loaders = _dataloader_split(dt, config['kfold'], config)

    return loaders


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


def _dataloader_split(dt, config):
    indices = list(range(len(dt)))
    np.random.shuffle(indices)

    kf = KFold(n_splits=config['kfold'], shuffle=False)
    indices_seq = list(kf.split(list(range(len(dt)))))
    ntest = len(indices_seq[0][1])

    loader_seq = []
    for trainval, test_indices in indices_seq:
        trainval = [indices[i] for i in trainval]
        test_indices = [indices[i] for i in test_indices]

        validation_indices = trainval[-ntest:]
        train_indices = trainval[:-ntest]

        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        validation_sampler = data.sampler.SubsetRandomSampler(validation_indices)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)

        train_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                       sampler=train_sampler,
                                       num_workers=config['num_workers'])
        validation_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                            sampler=validation_sampler,
                                            num_workers=config['num_workers'])
        test_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                      sampler=test_sampler,
                                      num_workers=config['num_workers'])

        loader_seq.append((train_loader, validation_loader, test_loader))
    return loader_seq


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
