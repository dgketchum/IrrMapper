import os
import numpy as np
from argparse import ArgumentParser

from matplotlib import pyplot as plt
from matplotlib import colors
from pytorch_lightning import Trainer

from data_preproc import feature_spec
from models.unet.unet import UNet
from models.nnet.nnet import NNet
from models.tcnn.tcnn import TempConv
from configure import get_config

FEATURES = feature_spec.features()
MODEL_MAP = {'unet': UNet,
             'nnet': NNet,
             'tcnn': TempConv}


def main(params):
    config = get_config(**vars(params))

    checkpoint_dir = os.path.join(params.checkpoint, 'checkpoints')
    figures_dir = os.path.join(params.checkpoint, 'figures')
    checkpoint = [os.path.join(checkpoint_dir, x) for x in os.listdir(checkpoint_dir)][0]
    print(checkpoint)

    model = MODEL_MAP[config.model]
    model = model.load_from_checkpoint(checkpoint)
    model.freeze()

    if 'nobackup' in model.hparams.dataset_folder:
        model.hparams.dataset_folder = '/media/nvm/ts_data/cm/images'

    model.hparams.batch_size = 1

    if params.metrics:
        trainer = Trainer(
            precision=16,
            gpus=config.device_ct,
            num_nodes=config.node_ct,
            log_every_n_steps=5)

        trainer.test(model)
        return

    loader = model.test_dataloader()

    for i, (x, g, y) in enumerate(loader):
        x, g, y, pred = model.predict_example(x, g, y)
        fig = os.path.join(figures_dir, '{}.png'.format(i))
        plot_prediction(x, geo=g, label=y, pred=pred, out_file=None)


def plot_prediction(x, geo=None, label=None, pred=None, out_file=None):
    cmap_label = colors.ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])
    cmap_pred = colors.ListedColormap(['blue', 'purple', 'pink', 'green'])

    r_idx, g_idx, b_idx, n_idx = [FEATURES.index(x) for x in FEATURES if 'red' in x], \
                                 [FEATURES.index(x) for x in FEATURES if 'green' in x], \
                                 [FEATURES.index(x) for x in FEATURES if 'blue' in x], \
                                 [FEATURES.index(x) for x in FEATURES if 'nir' in x]

    ndvi = np.max(((x[n_idx, :, :] - x[r_idx, :, :]) / (x[n_idx, :, :] + x[r_idx, :, :]) + 1e-5), axis=0)

    def to_rgb(arr):
        arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
        return arr

    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 10))
    r_idx, g_idx, b_idx = r_idx[3:10], g_idx[3:10], b_idx[3:10]
    r, g, b = x[r_idx, :, :], x[g_idx, :, :], x[b_idx, :, :]
    rgb = map(to_rgb, [np.median(r, axis=0), np.median(g, axis=0), np.median(b, axis=0)])

    rgb = np.dstack(list(rgb))

    lat = geo[0, :, :].mean()
    lon = geo[1, :, :].mean()

    print(lat, lon)

    ax[0].imshow(rgb)
    ax[0].set(xlabel='image')

    ax[1].imshow(ndvi, cmap='RdYlGn')
    ax[1].set(xlabel='ndvi_max')

    ax[2].imshow(label, cmap=cmap_label)
    ax[2].set(xlabel='label {}'.format(np.unique(label)))

    ax[3].imshow(pred, cmap=cmap_pred)
    ax[3].set(xlabel='pred {}'.format(np.unique(pred)))

    plt.suptitle('{:.3f}, {:.3f}'.format(lat, lon))
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # TODO: check nnet predictions
    # TODO: inv norm tcnn loader
    project = '/home/dgketchum/PycharmProjects/IrrMapper'
    checkpoint_pth = os.path.join(project, 'models/tcnn/results/pc-2021.04.06.14.11-tcnn-pixel')

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model', default='tcnn')
    parser.add_argument('--gpu', default='RTX')
    parser.add_argument('--machine', default='pc')
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--progress', default=0, type=int)
    parser.add_argument('--workers', default=6, type=int)
    parser.add_argument('--stack', default='cm')
    parser.add_argument('--checkpoint', default=checkpoint_pth)
    parser.add_argument('--metrics', default=False, type=bool)

    args = parser.parse_args()
    main(args)
# ========================= EOF ====================================================================
