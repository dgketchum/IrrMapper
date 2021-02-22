import os
import pickle as pkl
import numpy as np
from copy import deepcopy

import torch
from matplotlib import pyplot as plt
from matplotlib import colors

from utils import recursive_todevice
from learning.metrics import get_conf_matrix, confusion_matrix_analysis
from models.model_init import get_model
from data_load.data_loader import get_predict_loader
from configure import get_config
from data_preproc import feature_spec

FEATURES = feature_spec.features()


def unnormalize(x, config):
    """ get original image data (1 X H x W x T x C ) ==> ( H x W x C )"""
    mean_std = pkl.load(open(config['norm'], 'rb'))
    x = x.squeeze()
    if config['predict_mode'] == 'image':
        x = x.permute(2, 3, 0, 1)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
    mean, std = torch.tensor(mean_std[0][:x.shape[-1]]), torch.tensor(mean_std[1][:x.shape[-1]])
    x = x.mul_(std).add_(mean)
    x = x.detach().numpy()
    return x


def predict(config):
    device = torch.device(config['device'])

    n_class = config['num_classes']
    confusion = torch.tensor(np.zeros((n_class, n_class))).to(device)

    val_loader = get_predict_loader(config)
    model = get_model(config)
    check_pt = torch.load(os.path.join(config['res_dir'], 'model.pth.tar'))
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(check_pt['state_dict'], strict=False)
    chkpt = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
    # for x in chkpt:
    #     l = check_pt['state_dict'][x].cpu().numpy().tolist()
    #     with open('{}.txt'.format(x), 'w') as f:
    #         f.write(str(l))
    model.to(device)
    optimizer.load_state_dict(check_pt['optimizer'])
    model.eval()

    for i, (x, y, g) in enumerate(val_loader):
        image = unnormalize(deepcopy(x), config)
        x = recursive_todevice(x, device)

        if config['model'] == 'clstm':
            y = y.squeeze().to(device)
            mask = (y.sum(0) > 0).flatten()
            y = y.argmax(0)
            y = y.reshape(1, y.shape[0], y.shape[1])
            y_flat = y.flatten()
            with torch.no_grad():
                out, att = model(x)
                pred = out[0][0]
                pred_img = torch.argmax(pred, dim=1)
                pred_flat = pred_img.flatten()

        else:
            y = y.squeeze().to(device)
            x = x.squeeze()
            mask = (y.sum(0) > 0).flatten()

            if config['model'] == 'nnet':
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
            elif config['model'] == 'tcnn':
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[3], x.shape[2])
            else:
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

            with torch.no_grad():
                pred, att = model(x)
                pred_flat = torch.argmax(pred, dim=1)
                pred_img = pred_flat.reshape((image.shape[0], image.shape[1]))
                y_flat = y.argmax(0).flatten()

        pred_img = pred_img.cpu().numpy()
        y = y.cpu().numpy()
        print(np.unique(pred_img))
        g = g.numpy()
        out_fig = os.path.join(config['res_dir'], 'figures', '{}.png'.format(i))
        plot_prediction(image, pred_img, y, geo=g, out_file=out_fig, config=config)

        confusion += get_conf_matrix(y_flat[mask], pred_flat[mask], config['num_classes'], device)

    _, overall = confusion_matrix_analysis(confusion)
    prec, rec, f1 = overall['precision'], overall['recall'], overall['f1-score']
    print(confusion)
    print('Precision {:.4f}, Recall {:.4f}, F1 {:.2f},'.format(prec, rec, f1))


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
    config = get_config('clstm')
    predict(config)
# ========================= EOF ====================================================================
