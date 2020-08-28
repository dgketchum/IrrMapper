import json
import os
import scipy.ndimage.measurements as mnts
import numpy as np
from numpy import zeros_like, array, sort, sum, where, nan, swapaxes, count_nonzero
from numpy import nanmean, iinfo, uint32, sqrt, save, isnan, all, mean, any, int16
from map.trainer.training_utils import make_test_dataset
import pickle as pkl

# dates are generic, dates of each year as below, but data is from many years
# the year of the data is not used in training, just date position
DATES = {0: '19860101',
         1: '19860131',
         2: '19860302',
         3: '19860401',
         4: '19860501',
         5: '19860531',
         6: '19860630',
         7: '19860730',
         8: '19860829',
         9: '19860928',
         10: '19861028',
         11: '19861127',
         12: '19861227'}

# see feature_spec.py for dict of bands, lat , lon, elev, label
CHANNELS = 7
BANDS = 91

structure = array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])


def write_pixel_set(out, recs, n_samples=10000):
    """Iterate through each image of feature bands and labels, find contiguous objects, extract
    pixel set from within the objects
    """
    dataset = make_test_dataset(recs, True).batch(1)
    count = 0
    nan_pix = 0
    nan_geom = 0
    invalid_pix = 0
    mean_, std_ = 0, 0
    M2 = 0
    label_dict, size_dict, geom = {}, {}, {}
    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy().squeeze()
        features = features.numpy().squeeze()

        bbox_slices = {}
        for i in range(labels.shape[2]):
            _class = i + 1
            lab = labels[:, :, i].copy()
            if lab.max():
                bbox_slices[i] = mnts.find_objects(mnts.label(lab, structure=structure)[0])
                for b in bbox_slices[i]:
                    lab_mask = np.repeat(lab[b][:, :, np.newaxis], features.shape[-1], axis=2)
                    nan_label = lab_mask.copy()
                    nan_label[:, :, :] = iinfo(uint32).min
                    c = where(lab_mask, features[b], nan_label)
                    c[c == iinfo(uint32).min] = nan

                    # geometric features (lat, lon, elev)
                    geo = list(nanmean(c[:, :, -3:], axis=(0, 1)))

                    # pse.shape = T x C x S
                    c = c[:, :, :BANDS]
                    c = c.reshape(c.shape[0] * c.shape[1], BANDS)
                    nan_mask = all(isnan(c), axis=1)
                    c = c[~nan_mask]
                    c = c.reshape(c.shape[0], len(DATES.keys()), CHANNELS)
                    c = swapaxes(c, 0, 2)
                    c = swapaxes(c, 0, 1)

                    if count_nonzero(isnan(geo)):
                        nan_geom += 1
                        continue

                    if count_nonzero(isnan(c)):
                        nan_pix += 1
                        continue

                    if any(c[:, 0, :] == 2.0):
                        invalid_pix += 1
                        continue

                    count += 1

                    # update mean and std
                    # mean_std.shape =  C x T
                    delta = nanmean(c, axis=2) - mean_
                    mean_ = mean_ + delta / count
                    delta2 = nanmean(c, axis=2) - mean_
                    M2 = M2 + delta * delta2
                    std_ = sqrt(M2 / (count - 1))

                    geom[count] = geo
                    label_dict[count] = _class
                    size_dict[count] = c.shape[2]
                    if count % 10000 == 0:
                        print('count: {}'.format(count))

                    save(os.path.join(out, 'DATA', '{}'.format(count)), c)

        # display_box(labels, bbox_slices)
        if count > n_samples:
            print('final pse shape: {}'.format(c.shape))
            print('count of pixel sets: {}'.format(count))
            print('mean: {}'.format(list(mean_[:, 6])))
            print('std: {}'.format(list(std_[:, 6])))
            print('nan arrays: {}'.format(nan_pix))
            print('nan geom: {}'.format(nan_geom))
            print('invalid (2.0) pixel values: {}'.format(invalid_pix))

            with open(os.path.join(out, 'S2-2017-T31TFM-meanstd.pkl'), 'wb') as handle:
                pkl.dump((mean_, std_), handle, protocol=pkl.HIGHEST_PROTOCOL)

            label_dict = {'label_4class': label_dict}
            with open(os.path.join(out, 'META', 'labels.json'), 'w') as file:
                file.write(json.dumps(label_dict, indent=4))

            with open(os.path.join(out, 'META', 'dates.json'), 'w') as file:
                file.write(json.dumps(DATES, indent=4))

            with open(os.path.join(out, 'META', 'sizes.json'), 'w') as file:
                file.write(json.dumps(size_dict, indent=4))

            with open(os.path.join(out, 'META', 'geomfeat.json'), 'w') as file:
                file.write(json.dumps(geom, indent=4))
            exit()


def display_box(labels, slices):
    """"Display bounding boxes with pyplot, pixel set will be extracted from within these boxes"""
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])

    for n in range(labels.shape[-1]):
        labels[:, :, n] *= n + 1

    labels = sum(labels, axis=-1)
    boxes = zeros_like(labels)

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(labels, cmap=cmap)
    for i, c in enumerate(slices.keys(), start=1):
        for s in slices[c]:
            boxes[s] = c + 1
    ax[1].imshow(boxes, cmap=cmap)
    plt.show()


def date_parser(filepath):
    """
    returns the date (as int) from the file name of an S2 image
    """
    filename = os.path.split(filepath)[-1]
    return int(str(filename).split('.')[0])


def list_extension(folder, extension='tif'):
    """
    Lists files in folder with the specified extension
    """
    return [f for f in os.listdir(folder) if str(f).endswith(extension)]


def get_dates(input_folder):
    tifs = list_extension(input_folder, '.tif')
    tifs = sort(tifs)
    dates = [int(t.replace('.tif', '')) for t in tifs]

    ndates = len(dates)

    date_index = dict(zip(dates, range(ndates)))

    print('{} dates found in input folder '.format(ndates))
    tifs = [os.path.join(input_folder, f) for f in tifs]

    return tifs, date_index


def prepare_output(output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'DATA'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'META'), exist_ok=True)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    pixel_sets = os.path.join(home, 'IrrigationGIS', 'pixel_sets')
    tf_recs = os.path.join(home, 'IrrigationGIS', 'tfrecords')

    write_pixel_set(pixel_sets, tf_recs, label_names=['CODE_GROUP'])

# ========================= EOF ====================================================================
