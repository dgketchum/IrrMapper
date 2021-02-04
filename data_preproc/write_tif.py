import os

import numpy as np

from pyproj import Proj, Transformer
import rasterio
from rasterio.transform import from_origin

from data_preproc import feature_spec
from tf_dataset import make_test_dataset

MODE = 'irr'
FEATURES_DICT = feature_spec.features_dict()
FEATURES = feature_spec.features()
step_, length_ = 7, len(FEATURES)

r_idx, g_idx, b_idx = [FEATURES.index(x) for x in FEATURES if 'red' in x], \
                      [FEATURES.index(x) for x in FEATURES if 'green' in x], \
                      [FEATURES.index(x) for x in FEATURES if 'blue' in x]

lat_idx = [FEATURES.index(x) for x in FEATURES if 'lat' in x][0]
lon_idx = [FEATURES.index(x) for x in FEATURES if 'lon' in x][0]

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])


def build_raster(recs, out_tif_dir, pattern='*gz', plot=False):
    """write geoTiff from TFRecord"""
    dataset = make_test_dataset(recs, pattern).batch(1)
    for j, (features, labels) in enumerate(dataset):

        labels = labels.numpy().squeeze()
        features = features.numpy().squeeze()
        cdl = features[:, :, -6]
        cconf = features[:, :, -5]

        for n in range(labels.shape[-1]):
            labels[:, :, n] *= n + 1
        labels = np.sum(labels, axis=-1)

        r, g, b = features[:, :, r_idx], features[:, :, g_idx], features[:, :, b_idx]

        norm = lambda arr: ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255))
        rgb = map(norm, [np.median(r, axis=2), np.median(g, axis=2), np.median(b, axis=2)])
        rgb = np.dstack(list(rgb)).astype('uint8')

        lat, lon = features[:, :, lat_idx].max(), features[:, :, lon_idx].min()
        if plot:
            fig, ax = plt.subplots(ncols=4)
            ax[0].imshow(rgb)
            ax[1].imshow(labels, cmap=cmap)
            ax[2].imshow(cdl)
            ax[3].imshow(cconf)
            plt.suptitle('{:.3f}, {:.3f}'.format(lat, lon))
            plt.show()

        x1, y1 = lon, lat
        transformer = Transformer.from_crs('epsg:4326', 'epsg:5071')
        lon, lat = transformer.transform(y1, x1)
        affine = from_origin(lon, lat, 30, 30)
        tif_name = os.path.join(out_tif_dir, '{}.tif'.format(j))

        meta = dict(driver='GTiff',
                    height=rgb.shape[0], width=rgb.shape[1],
                    count=3, dtype=str(rgb.dtype),
                    crs='epsg:5071',
                    transform=affine)

        with rasterio.open(tif_name, 'w', **meta) as dst:
            for b in range(rgb.shape[-1]):
                band = rgb[:, :, b].astype(rasterio.uint8)
                dst.write_band(b + 1, band)


if __name__ == '__main__':
    records = '/media/hdisk/t_data/valid'
    tif = '/home/dgketchum/Downloads/tif'
    build_raster(records, tif, plot=False)
# ========================= EOF ====================================================================
