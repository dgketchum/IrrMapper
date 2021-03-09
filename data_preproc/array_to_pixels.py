import os
import numpy as np
import pickle as pkl
import webdataset as wds
import tarfile
import tempfile
import torch
import shutil

FEATURES = 96

# not incl background (0) class
N_CLASSES = 3


def push_tar(t_dir, out_dir, mode, items, ind):
    tar_filename = '{}_{}.tar'.format(mode, str(ind).zfill(6))
    tar_archive = os.path.join(out_dir, tar_filename)
    with tarfile.open(tar_archive, 'w') as tar:
        for i in items:
            tar.add(i, arcname=os.path.basename(i))
    shutil.rmtree(t_dir)


def write_pixel_blocks(data_dir, out, mode, n_subset=100, out_norm=None):
    """ write tensorflow records to class-balanced torch arrays every n samples"""

    label_count, pth_count, out_pixels = 0, 0, 0
    mean_, std_ = 0, 0
    M2 = 0
    obj_cntr = {1: None, 2: None, 3: None}
    pix_ct = np.array([0, 0, 0])

    url_list = [[os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith('.tar')]]

    for urls in url_list:

        dataset = wds.Dataset(urls).decode('torchl')
        items, tar_count, = [], 0
        tmpdirname = tempfile.mkdtemp()

        for j, f in enumerate(dataset):
            a = f['pth'].numpy()
            labels = a[:, :, 98].reshape(a.shape[0] * a.shape[1])
            features = a[:, :, :98].reshape(a.shape[0] * a.shape[1], 98)
            mask = labels > 0
            labels = labels[mask]
            features = features[mask, :]

            if not np.isfinite(features).all():
                print('image {} has nan/inf'.format(f))
                continue

            classes = np.array([np.count_nonzero(labels == i) for i in obj_cntr.keys()])

            labels = labels.reshape(labels.shape[0], 1)
            pixel_stack = np.append(features, labels, axis=1)
            pix_ct += classes

            delta = np.nanmean(features, axis=0) - mean_
            mean_ = mean_ + delta / label_count
            delta2 = np.nanmean(features, axis=0) - mean_
            M2 = M2 + delta * delta2
            std_ = np.sqrt(M2 / label_count)

            for k, v in obj_cntr.items():
                if isinstance(obj_cntr[k], np.ndarray):
                    obj_cntr[k] = np.append(obj_cntr[k], pixel_stack[pixel_stack[:, 98] == k], axis=0)
                else:
                    obj_cntr[k] = pixel_stack[pixel_stack[:, 98] == k]

            obj_state = np.array([isinstance(obj_cntr[x], type(None)) for x in obj_cntr.keys()])
            if obj_state.any():
                continue

            counts = np.array([obj_cntr[x].shape[0] > n_subset for x in obj_cntr.keys()])
            if counts.all():
                indices = [np.array([x for x in range(0, obj_cntr[y].shape[0])]) for y in obj_cntr.keys()]
                choice = [np.random.choice(i, n_subset, replace=False) for i in indices]
                choice_inv = [[i for i in ind if i not in choice[x]] for x, ind in enumerate(indices)]
                select = [obj_cntr[x][c, :] for x, c in zip(obj_cntr.keys(), choice)]
                left_over = {x: obj_cntr[x][ac, :] for x, ac in zip(obj_cntr.keys(), choice_inv)}

                array = np.concatenate(select, axis=0)
                out_pixels += array.shape[0]
                array = torch.from_numpy(array)
                tmp_tensor = os.path.join(tmpdirname, '{}.pth'.format(str(label_count).rjust(7, '0')))
                pth_count += 1
                torch.save(array, tmp_tensor)
                items.append(tmp_tensor)

                for k, v in obj_cntr.items():
                    if v.shape[0] > 5 * n_subset:
                        left_over[k] = v[:n_subset * 5, :]
                    obj_cntr[k] = left_over[k]

            if len(items) == 200:
                push_tar(tmpdirname, out, mode, items, tar_count)
                tmpdirname = tempfile.mkdtemp()
                items = []
                tar_count += 1

    if len(items) > 0:
        push_tar(tmpdirname, out, mode, items, tar_count)

    print('\n{}'.format(mode.upper()))
    print('pixel count: {}, {} total'.format(pix_ct, out_pixels))

    if out_norm:
        pkl_name = os.path.join(out_norm, 'meanstd.pkl')
        with open(pkl_name, 'wb') as handle:
            pkl.dump((mean_, std_), handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    data = '/media/hdisk/ts_data'

    images_dir = os.path.join(data, 'images')
    pixels_dir = os.path.join(data, 'pixels')

    for split in ['valid', 'test', 'train']:
        np_images = os.path.join(images_dir, split)
        pixel_dst = os.path.join(pixels_dir, split)
        if split == 'train':
            write_pixel_blocks(data_dir=np_images, out=pixel_dst, mode=split, out_norm=pixels_dir)
        else:
            write_pixel_blocks(data_dir=np_images, out=pixel_dst, mode=split)

# ========================= EOF ================================================================
