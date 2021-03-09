import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

from data_preproc import feature_spec

FEATURES = feature_spec.features()
R, G, B = [FEATURES.index(x) for x in FEATURES if 'red' in x], \
          [FEATURES.index(x) for x in FEATURES if 'green' in x], \
          [FEATURES.index(x) for x in FEATURES if 'blue' in x]
lat_idx = [FEATURES.index(x) for x in FEATURES if 'lat' in x][0]
lon_idx = [FEATURES.index(x) for x in FEATURES if 'lon' in x][0]


def plot_image_data(x, label=None, out_file=None):
    bounds = [0, 1, 2, 3, 4]
    cmap_label = colors.ListedColormap(['grey', 'blue', 'pink', 'green'])
    norm = colors.BoundaryNorm(bounds, len(bounds))

    r_idx, g_idx, b_idx = R[3:10], G[3:10], B[3:10]
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    r, g, b = x[:, :, r_idx], x[:, :, g_idx], x[:, :, b_idx]

    def norm_rgb(arr):
        arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
        return arr

    rgb = map(norm_rgb, [np.median(r, axis=2), np.median(g, axis=2), np.median(b, axis=2)])
    rgb = np.dstack(list(rgb))

    lat, lon = x[:, :, lat_idx].mean(), x[:, :, lon_idx].mean()

    ax[0].imshow(rgb)
    ax[0].set(xlabel='image')

    ax[1].imshow(label, cmap=cmap_label, norm=norm)
    ax[1].set(xlabel='label')

    plt.suptitle('{:.3f}, {:.3f} {}'.format(lat, lon, np.unique(label)))
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
