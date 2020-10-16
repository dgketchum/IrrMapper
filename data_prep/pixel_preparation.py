import os
import scipy.ndimage.measurements as mnts
import numpy as np
import pickle as pkl
import webdataset as wds
import tarfile
import tempfile
import torch
import shutil
from data_prep import BANDS, TERRAIN

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


def write_pixel_sets(out, recs, mode, out_norm):
    """Iterate through each image of feature bands and labels, find contiguous objects, extract
    pixel set from within the objects
    """

    count = 0
    invalid_pix, nan_pix = 0, 0
    mean_, std_ = 0, 0
    M2 = 0
    obj_ct = {0: 0, 1: 0, 2: 0, 3: 0}
    pix_ct = {0: 0, 1: 0, 2: 0, 3: 0}
    end_idx = len(os.listdir(recs)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(recs, brace_str)
    dataset = wds.Dataset(url).decode('torchl')
    tar_count, items = 0, []
    tmpdirname = tempfile.mkdtemp()
    for j, f in enumerate(dataset):
        a = f['pth'].numpy()
        labels = a[:, :, 98:]
        features = a[:, :, :98]

        bbox_slices = {}
        for i in range(labels.shape[2]):
            _class = i
            lab = labels[:, :, i].copy()
            if lab.max():
                bbox_slices[i] = mnts.find_objects(mnts.label(lab, structure=structure)[0])
                for b in bbox_slices[i]:

                    # features
                    lab_mask = np.repeat(lab[b][:, :, np.newaxis], features.shape[-1], axis=2)
                    nan_label = lab_mask.copy()
                    nan_label[:, :, :] = np.iinfo(np.uint32).min
                    feat = np.where(lab_mask, features[b], nan_label)
                    feat[feat == np.iinfo(np.uint32).min] = np.nan

                    feat = feat[:, :, :BANDS + TERRAIN]
                    feat = feat.reshape(feat.shape[0] * feat.shape[1], BANDS + TERRAIN)
                    nan_mask = np.any(np.isnan(feat), axis=1)
                    feat = feat[~nan_mask]

                    if feat.shape[0] < 1:
                        nan_pix += 1
                        continue

                    if np.any(feat[:, 0] == 2.0):
                        invalid_pix += 1
                        continue

                    count += 1

                    label = np.ones((feat.shape[0], 1)) * _class

                    # update mean and std
                    # mean_std.shape =  C x T
                    delta = np.nanmean(feat, axis=0) - mean_
                    mean_ = mean_ + delta / count
                    delta2 = np.nanmean(feat, axis=0) - mean_
                    M2 = M2 + delta * delta2
                    std_ = np.sqrt(M2 / count)

                    obj_ct[_class] += 1
                    pix_ct[_class] += feat.shape[0]

                    if count % 1000 == 0:
                        print('count: {}'.format(count))

                    tmp_tensor = os.path.join(tmpdirname, '{}.pth'.format(str(count).rjust(7, '0')))

                    out_array = np.append(feat, label, axis=1)
                    feat = torch.from_numpy(out_array)
                    torch.save(feat, tmp_tensor)

                    items.append(tmp_tensor)

                    if len(items) == 200:
                        push_tar(tmpdirname, out, mode, items, tar_count)
                        tmpdirname = tempfile.mkdtemp()
                        items = []
                        tar_count += 1

    if len(items) > 0:
        push_tar(tmpdirname, out, mode, items, tar_count)

    print('objects count: {}'.format(obj_ct))
    print('pixel count: {}'.format(pix_ct))
    print('nan pixels: {}, 2.0 pixels: {}'.format(nan_pix, invalid_pix))
    print('count of pixel sets: {}'.format(count))

    if out_norm:
        pkl_name = os.path.join(out_norm, 'meanstd.pkl')
        with open(pkl_name, 'wb') as handle:
            pkl.dump((mean_, std_), handle, protocol=pkl.HIGHEST_PROTOCOL)


def write_pixel_blocks(data_dir, out, mode, n_subset=10000):
    """ write numpy arrays every n samples from pixel sets"""

    end_idx = len(os.listdir(data_dir)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(data_dir, brace_str)
    dataset = wds.Dataset(url).decode('torchl').to_tuple('pth')
    first = True
    count, file_count = 0, 0
    remainder = np.array(False)
    tmpdirname = tempfile.mkdtemp()
    items, tar_count = [], 0

    for sample in dataset:
        a = sample[0].numpy()
        if first:
            features = a
            first = False
            if remainder.any():
                features = np.append(features, remainder, axis=0)
        else:
            features = np.append(features, a, axis=0)
        if features.shape[0] > n_subset:
            remainder = features[n_subset:, :]
            out_features = torch.tensor(features[:n_subset, :])
            tmp_feat = os.path.join(tmpdirname, '{}.pth'.format(str(file_count).rjust(7, '0')))

            torch.save(out_features, tmp_feat)

            file_count += 1
            features = None
            count += 1
            first = True
            items.append(tmp_feat)

            if len(items) == 4:
                push_tar(tmpdirname, out, mode, items, tar_count)
                tmpdirname = tempfile.mkdtemp()
                items = []
                tar_count += 1

    if features is not None:
        while features.shape[-1] < n_subset:
            features = np.append(features, features, axis=-1)
        file_count += 1
        out_features = torch.tensor(features[:n_subset, :])
        tmp_feat = os.path.join(tmpdirname, '{}.pth'.format(str(file_count).rjust(7, '0')))

        torch.save(out_features, tmp_feat)

        print('final file {}'.format(file_count))
        items.append(tmp_feat)
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

        # if split == 'train':
        #     np_images = os.path.join(images, split, '{}_points'.format(split))
        #     pixel_dst = os.path.join(pixels, split, '{}_points'.format(split))
        #     pixel_set_dst = os.path.join(pixel_sets, split, '{}_points'.format(split))

        if split == 'train':
            out_norm_pse = pixel_sets
            out_norm_pixels = pixels
        else:
            out_norm_pixels = None
            out_norm_pse = None

        # write_pixel_sets(pixel_set_dst, np_images, split, out_norm=out_norm_pse)
        write_pixel_blocks(pixel_set_dst, pixel_dst, split)

# ========================= EOF ================================================================
