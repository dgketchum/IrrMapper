import json
import os
import scipy.ndimage.measurements as mnts
import numpy as np
import pickle as pkl
import webdataset as wds
import tarfile
import tempfile
import torch
import shutil

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


def push_tar(t_dir, out_dir, mode, items, ind):
    tar_filename = '{}_{}.tar'.format(mode, str(ind).zfill(6))
    tar_archive = os.path.join(out_dir, tar_filename)
    with tarfile.open(tar_archive, 'w') as tar:
        for i in items:
            tar.add(i, arcname=os.path.basename(i))
    shutil.rmtree(t_dir)


def write_pixel_sets(out, recs, mode):
    """Iterate through each image of feature bands and labels, find contiguous objects, extract
    pixel set from within the objects
    """

    count = 0
    nan_pix = 0
    nan_geom = 0
    invalid_pix = 0
    mean_, std_ = 0, 0
    M2 = 0
    obj_ct = {0: 0, 1: 0, 2: 0, 3: 0}
    end_idx = len(os.listdir(recs)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(recs, brace_str)
    dataset = wds.Dataset(url).decode('torchl')
    tar_count, items = 0, []
    tmpdirname = tempfile.mkdtemp()
    for j, f in enumerate(dataset):
        a = f['pth'].numpy()
        labels = a[:, :, 98:]
        features = a[:, :, :96]

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

                    ancillary = {'label': _class,
                                 'size': feat.shape[2],
                                 'terrain': geo}

                    if count % 100 == 0:
                        print('count: {}'.format(count))

                    tmp_tensor = os.path.join(tmpdirname, '{}.pth'.format(str(count).rjust(7, '0')))
                    tmp_json = os.path.join(tmpdirname, '{}.json'.format(str(count).rjust(7, '0')))

                    feat = torch.from_numpy(feat)
                    torch.save(feat, tmp_tensor)
                    with open(tmp_json, 'w') as file:
                        file.write(json.dumps(ancillary, indent=4))

                    items.extend([tmp_tensor, tmp_json])

                    if len(items) == 200:
                        push_tar(tmpdirname, out, mode, items, tar_count)
                        tmpdirname = tempfile.mkdtemp()
                        items = []
                        tar_count += 1

    print('objects count: {}'.format(obj_ct))
    print('final pse shape: {}'.format(feat.shape))
    print('count of pixel sets: {}'.format(count))
    print('nan arrays: {}'.format(nan_pix))
    print('nan geom: {}'.format(nan_geom))
    print('invalid (2.0) pixel values: {}'.format(invalid_pix))

    pkl_name = os.path.join(os.path.dirname(out), '{}_meanstd.pkl'.format(mode))
    with open(pkl_name, 'wb') as handle:
        pkl.dump((mean_, std_), handle, protocol=pkl.HIGHEST_PROTOCOL)
    push_tar(tmpdirname, out, mode, items, tar_count)


def write_pixel_blocks(data_dir, out, mode, n_subset=1000):
    """ write numpy arrays every n samples from pixel sets"""

    end_idx = len(os.listdir(data_dir)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(data_dir, brace_str)
    dataset = wds.Dataset(url).decode('torchl').to_tuple('pth', 'json')
    first = True
    count, file_count = 0, 0
    features, labels = None, []
    remainder, remainder_lab = np.array(False), []
    tmpdirname = tempfile.mkdtemp()
    items, tar_count = [], 0

    for sample in dataset:
        a = sample[0].numpy()
        labels.extend([sample[1]['label'] for _ in range(a.shape[-1])])
        if first:
            features = a
            first = False
            if remainder.any():
                features = np.append(features, remainder, axis=-1)
                labels.extend(remainder_lab)
        else:
            features = np.append(features, a, axis=-1)
        if features.shape[-1] > n_subset:
            remainder, remainder_lab = features[:, :, n_subset:], labels[n_subset:]
            out_features, out_labels = torch.tensor(features[:, :, :n_subset]), {'labels': labels[:n_subset]}
            tmp_feat = os.path.join(tmpdirname, '{}.pth'.format(str(file_count).rjust(7, '0')))
            tmp_lab = os.path.join(tmpdirname, '{}.json'.format(str(file_count).rjust(7, '0')))

            torch.save(out_features, tmp_feat)
            with open(tmp_lab, 'w') as file:
                file.write(json.dumps(out_labels, indent=4))

            file_count += 1
            print('file {}, labels: {}, data {}'.format(file_count, out_features.shape[-1], len(out_labels)))
            print('remainder', remainder.shape, len(remainder_lab))
            print('')
            features, labels = None, []
            count += 1
            first = True
            items.extend([tmp_feat, tmp_lab])

            if len(items) == 4:
                push_tar(tmpdirname, out, mode, items, tar_count)
                tmpdirname = tempfile.mkdtemp()
                items = []
                tar_count += 1

    while features.shape[-1] < n_subset:
        features = np.append(features, features, axis=-1)
        labels.extend(labels)
    file_count += 1
    out_features, out_labels = torch.tensor(features[:, :, :n_subset]), {'labels': labels[:n_subset]}
    tmp_feat = os.path.join(tmpdirname, '{}.pth'.format(str(file_count).rjust(7, '0')))
    tmp_lab = os.path.join(tmpdirname, '{}.json'.format(str(file_count).rjust(7, '0')))

    torch.save(out_features, tmp_feat)
    with open(tmp_lab, 'w') as file:
        file.write(json.dumps(out_labels, indent=4))

    print('final file {}, labels: {}, data {}'.format(file_count, out_features.shape[-1], len(out_labels)))
    items.extend([tmp_feat, tmp_lab])
    push_tar(tmpdirname, out, mode, items, tar_count)


if __name__ == '__main__':
    data = '/home/dgketchum/IrrigationGIS/tfrecords/tarchives'

    if not os.path.isdir(data):
        data = '/mnt/beegfs/dk128872/ts_data/cmask/tar'

    images = os.path.join(data, 'images')
    pixels = os.path.join(data, 'pixels')
    pixel_sets = os.path.join(data, 'pixel_sets')

    for split in ['train', 'test', 'valid']:
        np_images = os.path.join(images, split, '{}_patches'.format(split))
        pixel_dst = os.path.join(pixels, split, '{}_patches'.format(split))
        pixel_set_dst = os.path.join(pixel_sets, split, '{}_patches'.format(split))

        # write_pixel_sets(pixel_set_dst, np_images, split)
        write_pixel_blocks(pixel_set_dst, pixel_dst, split)

# ========================= EOF ================================================================
