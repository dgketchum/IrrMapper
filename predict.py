import os
import pickle as pkl

import torch
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from utils import recursive_todevice
from models.model_init import get_model, get_predict_loader
from configure import get_config
from data_prep import FEATURES

cmap = ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])


def unnormalize(x, config):
    """ get original image data"""
    x = x.squeeze()
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
    x = x.permute(1, 2, 0)
    mean_std = torch.tensor(pkl.load(open(config['dataset_folder'] + '/meanstd.pkl', 'rb')))
    mean, std = mean_std[0], mean_std[1]
    mean = mean.reshape(mean.shape[0] * mean.shape[1])
    std = std.reshape(std.shape[0] * std.shape[1])
    x = x.mul_(std).add_(mean)
    x = x.detach().numpy()
    return x


def predict(config):
    device = torch.device(config['device'])
    val_loader = get_predict_loader(config)
    print('Predict')
    model = get_model(config)
    check_pt = torch.load(os.path.join(config['res_dir'], 'model.pth.tar'))
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(check_pt['state_dict'])
    model.to(device)
    optimizer.load_state_dict(check_pt['optimizer'])
    model.eval()
    for i, (x, y) in enumerate(val_loader):
        image = unnormalize(x, config)
        x = recursive_todevice(x, device)
        if config['model'] == 'clstm':
            y = y.argmax(dim=1).to(device)
            with torch.no_grad():
                out, att = model(x)
                pred = out[0][0]
                pred = torch.argmax(pred, dim=1)
        else:
            y = y.to(device)
            y = y.argmax(dim=1)
            x = x.squeeze()
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            x = x.permute(2, 0, 1)
            with torch.no_grad():
                pred, att = model(x)
                pred = torch.argmax(pred, dim=1)
                pred = pred.reshape(y.shape)

        plot_prediction(image, pred.cpu().numpy(), y.cpu().numpy(), None)
        # os.path.join(config['res_dir'], 'figures', '{}.png'.format(i)))


def plot_prediction(x, pred, label, out_file=None):
    cmap = ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])
    r_idx, g_idx, b_idx = [FEATURES.index(x) for x in FEATURES if 'red' in x], \
                          [FEATURES.index(x) for x in FEATURES if 'green' in x], \
                          [FEATURES.index(x) for x in FEATURES if 'blue' in x]

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
    print(x.shape)
    r, g, b = x[:, :, r_idx], x[:, :, g_idx], x[:, :, b_idx]

    def norm_rgb(arr):
        arr = (arr * 255).astype('uint8')
        return arr

    rgb = map(norm_rgb, [np.median(r, axis=2), np.median(g, axis=2), np.median(b, axis=2)])
    rgb = np.dstack(rgb)

    lat, lon = x[:, :, -3].mean(), x[:, :, -2].mean()

    pred, label = pred.squeeze(), label.squeeze()
    ax[0].imshow(rgb)
    # ax[1].imshow(label, cmap=cmap)
    # ax[2].imshow(pred, cmap=cmap)
    plt.suptitle('{:.3f}, {:.3f}'.format(lat, lon))
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()
        exit()


if __name__ == '__main__':
    config = get_config('tcnn')
    predict(config)
# ========================= EOF ====================================================================
