import pickle as pkl
import numpy as np
from argparse import ArgumentParser

from matplotlib import pyplot as plt
from matplotlib import colors
from pytorch_lightning import Trainer

from data_preproc import feature_spec
from models.unet.unet import UNet
from configure import get_config

FEATURES = feature_spec.features()


def main(params):
    config = get_config(**vars(params))

    model = UNet.load_from_checkpoint(checkpoint_path=params.checkpoint)

    trainer = Trainer(
        precision=16,
        gpus=config.device_ct,
        num_nodes=config.node_ct,
        log_every_n_steps=5)

    trainer.test(model)


def plot_prediction(x, pred=None, label=None, geo=None, out_file=None, config=None):
    cmap_label = colors.ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])
    cmap_pred = colors.ListedColormap(['blue', 'purple', 'pink', 'green'])

    r_idx, g_idx, b_idx = [FEATURES.index(x) for x in FEATURES if 'red' in x], \
                          [FEATURES.index(x) for x in FEATURES if 'green' in x], \
                          [FEATURES.index(x) for x in FEATURES if 'blue' in x]

    r_idx, g_idx, b_idx = r_idx[3:10], g_idx[3:10], b_idx[3:10]
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
    r, g, b = x[:, :, r_idx], x[:, :, g_idx], x[:, :, b_idx]

    def norm_rgb(arr):
        arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
        return arr

    rgb = map(norm_rgb, [np.median(r, axis=2), np.median(g, axis=2), np.median(b, axis=2)])
    rgb = np.dstack(list(rgb))

    geo = geo.squeeze()
    mean_std = pkl.load(open(config['norm'], 'rb'))
    lat_std, lat_mn = mean_std[1][91], mean_std[0][91]
    lon_std, lon_mn = mean_std[1][92], mean_std[0][92]
    lat = geo[0, :, :].mean() * lat_std + lat_mn
    lon = geo[1, :, :].mean() * lon_std + lon_mn

    print(lat, lon)

    mask = label.sum(0) == 0
    label = label.argmax(0) + 1
    label[mask] = 0
    pred += 1
    pred, label = pred.squeeze(), label.squeeze()

    ax[0].imshow(rgb)
    ax[0].set(xlabel='image')

    ax[1].imshow(label, cmap=cmap_label)
    ax[1].set(xlabel='label')

    ax[2].imshow(pred, cmap=cmap_pred)
    ax[2].set(xlabel='pred')

    plt.suptitle('{:.3f}, {:.3f}'.format(lat, lon))
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    checkpoint_pth = '/home/dgketchum/PycharmProjects/IrrMapper/models/unet/results/' \
                     'pc-2021.03.22.15.03-unet-image/checkpoints/epoch=71-step=8135.ckpt'
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model', default='unet')
    parser.add_argument('--mode', default='image')
    parser.add_argument('--gpu', default='RTX')
    parser.add_argument('--machine', default='pc')
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--progress', default=0, type=int)
    parser.add_argument('--workers', default=6, type=int)
    parser.add_argument('--checkpoint', default=checkpoint_pth)
    args = parser.parse_args()
    main(args)
# ========================= EOF ====================================================================
