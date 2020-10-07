import json
import os
import scipy.ndimage.measurements as mnts
import numpy as np
import pickle as pkl
import webdataset as wds
import torch

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

structure = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])


def write_pixel_sets(out, recs, mode):
    """Iterate through each image of feature bands and labels, find contiguous objects, extract
    pixel set from within the objects
    """
    if not os.path.isdir(os.path.join(out, 'DATA')):
        os.mkdir(os.path.join(out, 'DATA'))
    if not os.path.isdir(os.path.join(out, 'META')):
        os.mkdir(os.path.join(out, 'META'))

    l = [os.path.join(recs, x) for x in os.listdir(recs)]
    count = 0
    nan_pix = 0
    nan_geom = 0
    invalid_pix = 0
    mean_, std_ = 0, 0
    M2 = 0
    # [757 128 329 986]
    obj_ct = {0: 0, 1: 0, 2: 0, 3: 0}
    label_dict, cdl_label_dict, size_dict, geom = {}, {}, {}, {}

    end_idx = len(os.listdir(recs)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(recs, brace_str)
    dataset = wds.Dataset(url).decode('torchl')

    for j, f in enumerate(dataset):
        a = f['pth'].numpy()
        labels = a[:, :, 98:]
        features = a[:, :, :96]
        cdl_label = a[:, :, 96:98]

        bbox_slices = {}
        for i in range(labels.shape[2]):
            _class = i
            lab = labels[:, :, i].copy()
            if lab.max():
                bbox_slices[i] = mnts.find_objects(mnts.label(lab, structure=structure)[0])
                for b in bbox_slices[i]:

                    # features
                    obj_ct[_class] += 1
                    lab_mask = np.repeat(lab[b][:, :, np.newaxis], features.shape[-1], axis=2)
                    nan_label = lab_mask.copy()
                    nan_label[:, :, :] = np.iinfo(np.uint32).min
                    feat = np.where(lab_mask, features[b], nan_label)
                    feat[feat == np.iinfo(np.uint32).min] = np.nan

                    # geometric features (lat, lon, elev, slope, aspect)
                    geo = list(np.nanmean(feat[:, :, 91:96], axis=(0, 1)))

                    # pse.shape = T x C x S
                    feat = feat[:, :, :BANDS]
                    feat = feat.reshape(feat.shape[0] * feat.shape[1], BANDS)
                    nan_mask = np.all(np.isnan(feat), axis=1)
                    feat = feat[~nan_mask]
                    feat = feat.reshape(feat.shape[0], len(DATES.keys()), CHANNELS)
                    feat = np.swapaxes(feat, 0, 2)
                    feat = np.swapaxes(feat, 0, 1)

                    # ignore cdl labels for now
                    # TODO: handle data so Irr and CDL are masked according to irrigation and CDL labels
                    # cdl label
                    # lab_mask = np.repeat(lab[b][:, :, np.newaxis], cdl_label.shape[-1], axis=2)
                    # nan_label = lab_mask.copy()
                    # nan_label[:, :, :] = np.iinfo(np.uint32).min
                    # cdl = np.where(lab_mask, cdl_label[b], nan_label)
                    # cdl[cdl == np.iinfo(np.uint32).min] = np.nan
                    #
                    # cdl = cdl.reshape(cdl.shape[0] * cdl.shape[1], 2)
                    # nan_mask = np.all(np.isnan(cdl), axis=1)
                    # cdl = cdl[~nan_mask]
                    # cdl = np.swapaxes(cdl, 0, 1)

                    if np.count_nonzero(np.isnan(geo)):
                        nan_geom += 1
                        continue

                    if np.count_nonzero(np.isnan(feat)):
                        nan_pix += 1
                        continue

                    if np.any(feat[:, 0, :] == 2.0):
                        invalid_pix += 1
                        continue

                    count += 1

                    # update mean and std
                    # mean_std.shape =  C x T
                    delta = np.nanmean(feat, axis=2) - mean_
                    mean_ = mean_ + delta / count
                    delta2 = np.nanmean(feat, axis=2) - mean_
                    M2 = M2 + delta * delta2
                    std_ = np.sqrt(M2 / (count - 1))

                    geom[count] = geo
                    label_dict[count] = _class
                    cdl_label_dict[count] = _class
                    size_dict[count] = feat.shape[2]
                    if count % 100 == 0:
                        print('count: {}'.format(count))

                    np.save(os.path.join(out, 'DATA', '{}'.format(count)), feat)
                    # np.save(os.path.join(out, 'DATA', '{}_cdl'.format(count)), cdl)

    print('objects count: {}'.format(obj_ct))
    print('final pse shape: {}'.format(feat.shape))
    print('count of pixel sets: {}'.format(count))
    print('nan arrays: {}'.format(nan_pix))
    print('nan geom: {}'.format(nan_geom))
    print('invalid (2.0) pixel values: {}'.format(invalid_pix))

    with open(os.path.join(out, 'meanstd.pkl'), 'wb') as handle:
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


def write_pixel_blocks(data_dir, out, n_subset=100000):
    """ write numpy arrays every n samples from pixel sets"""

    np_data = os.path.join(data_dir, 'DATA')
    meta_data = os.path.join(data_dir, 'META')
    _files = sorted([os.path.join(np_data, x) for x in os.listdir(np_data) if not x.endswith('_cdl.npy')])
    with open(os.path.join(meta_data, 'labels.json'), 'r') as file:
        label = json.loads(file.read())['label_4class']

    first = True
    count = 0
    start_ind, end_ind = 0, 0
    features, labels = None, []
    for j, (f, l) in enumerate(zip(_files, label)):
        a = np.load(f)
        # ignore cdl labels for now
        # TODO: handle data so Irr and CDL are masked according to irrigation and CDL labels
        # c = np.load(f.replace('.npy', '_cdl.npy'))
        # labels.extend([(label[str(j + 1)], c[1, i], c[0, i]) for i in range(a.shape[-1])])
        labels.extend([label[str(j + 1)] for i in range(a.shape[-1])])
        if first:
            features = a
            start_ind = count
            first = False
        else:
            features = np.append(features, a, axis=-1)
        count += a.shape[-1]
        if features.shape[-1] > n_subset:
            end_ind = count
            np.save(os.path.join(out, '{}-{}'.format(start_ind, end_ind)), features)
            np.save(os.path.join(out, '{}-{}_labels'.format(start_ind, end_ind)), np.array(labels))
            print('{}-{}'.format(start_ind, end_ind))
            features, labels = None, []
            count += 1
            first = True
    print(count)


if __name__ == '__main__':
    data = '/home/dgketchum/IrrigationGIS/tfrecords'
    tar = os.path.join(data, 'tarchives')

    if not os.path.isdir(data):
        data = '/mnt/beegfs/dk128872/ts_data/cmask'
        tar = os.path.join(data, 'tar')

    pixels = os.path.join(data, 'pixels')
    pixel_sets = os.path.join(data, 'pixel_sets')

    for split in ['train', 'test', 'valid']:
        np_images = os.path.join(tar, split, '{}_patches'.format(split))
        pixel_dst = os.path.join(pixels, split, '{}_patches'.format(split))
        pixel_set_dst = os.path.join(pixel_sets, split, '{}_patches'.format(split))

        write_pixel_sets(pixel_set_dst, np_images, split)
        write_pixel_blocks(pixel_set_dst, pixel_dst)

# ========================= EOF ================================================================
