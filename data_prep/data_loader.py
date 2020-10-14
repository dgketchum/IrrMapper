import os
import pickle as pkl

from torch.utils import data as data

from data_prep.dataset import image_dataset, pixel_dataset, pixelset_dataset


def get_predict_loader(config):
    mean_std = pkl.load(open(config['image_norm'], 'rb'))
    dt = image_dataset('test', config, mean_std, pixel_predict=True)
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
        dt = (pixelset_dataset(split, config, mean_std) for split in splits)

    if config['model'] == 'clstm':
        mean_std = pkl.load(open(config['image_norm'], 'rb'))
        dt = (image_dataset(split, config, mean_std, False) for split in splits)

    train, test, valid = (get_dataloader(d, config) for d in dt)
    return train, test, valid


def get_dataloader(dt, config):
    loader = data.DataLoader(dt, batch_size=config['batch_size'],
                             num_workers=config['num_workers'], pin_memory=False)
    return loader


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
