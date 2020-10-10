import os
import pickle as pkl
import numpy as np
from copy import deepcopy

import torch
from matplotlib import pyplot as plt
from matplotlib import colors

from utils import recursive_todevice
from models.model_init import get_model, get_predict_loader
from configure import get_config
from data_prep import FEATURES


def unnormalize(x, mean_std):
    """ get original image data"""
    x = x.squeeze()
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
    mean, std = torch.tensor(mean_std[0]), torch.tensor(mean_std[1])
    x = x.mul_(std).add_(mean)
    x = x.detach().numpy()
    return x


def predict(config):
    print('Predict')
    device = torch.device(config['device'])
    val_loader = get_predict_loader(config)
    norm = pkl.load(open(config['image_norm'], 'rb'))
    model = get_model(config)
    check_pt = torch.load(os.path.join(config['res_dir'], 'model.pth.tar'))
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(check_pt['state_dict'])
    model.to(device)
    optimizer.load_state_dict(check_pt['optimizer'])
    model.eval()
    for i, (x, y, g) in enumerate(val_loader):
        image = unnormalize(deepcopy(x), norm)
        x = recursive_todevice(x, device)
        if config['model'] == 'clstm':
            y = y.argmax(dim=1).numpy()
            with torch.no_grad():
                out, att = model(x)
                pred = out[0][0]
                pred = torch.argmax(pred, dim=1)
                pred = pred.cpu().numpy()
                g = g.numpy()
        else:

            y = y.squeeze().numpy()
            x = x.squeeze()
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
            with torch.no_grad():
                pred, att = model(x)
                pred = torch.argmax(pred, dim=1)
                pred = pred.reshape((image.shape[0], image.shape[1]))
                pred = pred.cpu().numpy()
                g = g.numpy()
        plot_prediction(image, pred, y, geo=g, out_file=os.path.join(config['res_dir'],
                                                                     'figures', '{}.png'.format(i)))


def plot_prediction(x, pred=None, label=None, geo=None, out_file=None):
    cmap = colors.ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])

    r_idx, g_idx, b_idx = [FEATURES.index(x) for x in FEATURES if 'red' in x], \
                          [FEATURES.index(x) for x in FEATURES if 'green' in x], \
                          [FEATURES.index(x) for x in FEATURES if 'blue' in x]

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
    r, g, b = x[:, :, r_idx], x[:, :, g_idx], x[:, :, b_idx]

    def norm_rgb(arr):
        arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
        return arr

    rgb = map(norm_rgb, [np.median(r, axis=2), np.median(g, axis=2), np.median(b, axis=2)])
    rgb = np.dstack(list(rgb))

    geo = geo.squeeze()
    lat, lon = geo[0, :, :].mean(), geo[1, :, :].mean()

    mask = label.sum(0) == 0
    label = label.argmax(0) + 1
    label[mask] = 0
    pred, label = pred.squeeze(), label.squeeze()
    print('pred', pred.min(), pred.max())
    print('label', label.min(), label.max(), label.mean())

    ax[0].imshow(rgb)
    ax[0].set(xlabel='image')

    ax[1].imshow(label, cmap=cmap)
    ax[1].set(xlabel='label')

    ax[2].imshow(pred, cmap=cmap)
    ax[2].set(xlabel='pred')

    plt.suptitle('{:.3f}, {:.3f}'.format(lat, lon))
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    config = get_config('tcnn')
    predict(config)
# ========================= EOF ====================================================================
