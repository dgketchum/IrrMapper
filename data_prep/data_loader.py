import pickle as pkl

from torch.utils import data as data

from data_prep.dataset import image_dataset, pixel_dataset, pixelset_dataset, predict_dataset


def get_predict_loader(config):

    mean_std = pkl.load(open(config['norm'], 'rb'))
    config['batch_size'] = 1

    if config['predict_mode'] == 'pixel':
        dt = predict_dataset('test', config, mean_std)
    elif config['predict_mode'] == 'image':
        dt = image_dataset('test', config, mean_std)
    else:
        dt = None
        raise NotImplementedError

    test = get_dataloader(dt, config)
    return test


def get_loaders(config):
    dt = None
    splits = ['train', 'test', 'valid']
    mean_std = pkl.load(open(config['norm'], 'rb'))

    if config['model'] in ['tcnn', 'dcm']:
        dt = (pixel_dataset(split, config, mean_std) for split in splits)

    if config['model'] == 'ltae':
        dt = (pixelset_dataset(split, config, mean_std) for split in splits)

    if config['model'] == 'clstm':
        dt = (image_dataset(split, config, mean_std) for split in splits)

    train, test, valid = (get_dataloader(d, config) for d in dt)
    return train, test, valid


def get_dataloader(dt, config):
    loader = data.DataLoader(dt, batch_size=None,
                             num_workers=config['num_workers'], pin_memory=False)
    return loader


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
