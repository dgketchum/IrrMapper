# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
import torch
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from data_prep.pixel_preparation import BANDS, CHANNELS, DATES

from models.model_init import get_model

sequence_len = len(DATES.keys())

features = ['blue_0', 'green_0', 'red_0', 'nir_0', 'swir1_0', 'swir2_0', 'tir_0', 'blue_1', 'green_1', 'red_1', 'nir_1',
            'swir1_1', 'swir2_1', 'tir_1', 'blue_2', 'green_2', 'red_2', 'nir_2', 'swir1_2', 'swir2_2', 'tir_2',
            'blue_3', 'green_3', 'red_3', 'nir_3', 'swir1_3', 'swir2_3', 'tir_3', 'blue_4', 'green_4', 'red_4', 'nir_4',
            'swir1_4', 'swir2_4', 'tir_4', 'blue_5', 'green_5', 'red_5', 'nir_5', 'swir1_5', 'swir2_5', 'tir_5',
            'blue_6', 'green_6', 'red_6', 'nir_6', 'swir1_6', 'swir2_6', 'tir_6', 'blue_7', 'green_7', 'red_7', 'nir_7',
            'swir1_7', 'swir2_7', 'tir_7', 'blue_8', 'green_8', 'red_8', 'nir_8', 'swir1_8', 'swir2_8', 'tir_8',
            'blue_9', 'green_9', 'red_9', 'nir_9', 'swir1_9', 'swir2_9', 'tir_9', 'blue_10', 'green_10', 'red_10',
            'nir_10', 'swir1_10', 'swir2_10', 'tir_10', 'blue_11', 'green_11', 'red_11', 'nir_11', 'swir1_11',
            'swir2_11', 'tir_11', 'blue_12', 'green_12', 'red_12', 'nir_12', 'swir1_12', 'swir2_12', 'tir_12', 'lat',
            'lon', 'elev']


def loader(path, norm):
    a = np.load(path)
    x = a[:, :, :BANDS]
    x = x.reshape(x.shape[0], x.shape[1], sequence_len, CHANNELS)
    x = x.transpose((2, 3, 0, 1))

    y = a[:, :, -4:]
    y = torch.from_numpy(y).permute(2, 0, 1)

    if norm is not None:
        m, s = norm
        m = np.array(m).reshape((sequence_len, CHANNELS, 1, 1))
        s = np.array(s).reshape((sequence_len, CHANNELS, 1, 1))
        x = (x - m) / s

    x = torch.from_numpy(x).float()
    return x, y


def predict_image(model, device, norm, config):
    r_idx, g_idx, b_idx = [features.index(x) for x in features if 'red' in x], \
                          [features.index(x) for x in features if 'green' in x], \
                          [features.index(x) for x in features if 'blue' in x]

    model = get_model(config)

    cmap = ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])
    x, y = loader('/home/dgketchum/PycharmProjects/IrrMapper/data/npy/969.npy', norm)
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        out, att = model(x)

    pred = out[0][0]
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    label = torch.sum(y, dim=1).cpu().numpy()
    r, g, b = x[:, :, r_idx], x[:, :, g_idx], x[:, :, b_idx]

    norm_rgb = lambda arr: ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
    rgb = map(norm_rgb, [np.median(r, axis=2), np.median(g, axis=2), np.median(b, axis=2)])
    rgb = np.dstack(rgb)

    lat, lon = x[:, :, -3].mean(), x[:, :, -2].mean()

    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(rgb)
    ax[1].imshow(label, cmap=cmap)
    ax[2].imshow(pred, cmap=cmap)
    plt.suptitle('{:.3f}, {:.3f}'.format(lat, lon))
    plt.show()
    pass


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
