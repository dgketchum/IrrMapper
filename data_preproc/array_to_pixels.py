import os
import numpy as np
import pickle as pkl
import torch

from configure import BANDS, TERRAIN, CDL

FEATURES = BANDS + TERRAIN + CDL
# not incl background (0) class


def write_pixel_blocks(data_dir, out, mode, n_subset=1000, out_norm=None):
    """ write tensorflow records to class-balanced torch arrays every n samples"""

    label_count, pth_count, out_pixels = 0, 0, 0
    mean_, std_ = 0, 0
    M2 = 0
    obj_cntr = {1: None, 2: None, 3: None}
    pix_ct = np.array([0, 0, 0])

    files = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith('.pth')]

    for j, f in enumerate(files):
        a = torch.load(f).numpy()
        labels = a[:, :, FEATURES].reshape(a.shape[0] * a.shape[1])
        features = a[:, :, :FEATURES].reshape(a.shape[0] * a.shape[1], 98)
        mask = labels > 0
        labels = labels[mask]
        features = features[mask, :]

        if not np.isfinite(features).all():
            print('image {} has nan/inf'.format(f))
            continue
        if not np.isfinite(labels).all():
            print('labels {} has nan/inf'.format(f))
            continue
        classes = np.array([np.count_nonzero(labels == i) for i in obj_cntr.keys()])

        labels = labels.reshape(labels.shape[0], 1)
        pixel_stack = np.append(features, labels, axis=1)
        pix_ct += classes

        if out_norm:
            delta = np.nanmean(features, axis=0) - mean_
            mean_ = mean_ + delta / (j + 1)
            delta2 = np.nanmean(features, axis=0) - mean_
            M2 = M2 + delta * delta2
            std_ = np.sqrt(M2 / (j + 1))

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
            np.random.shuffle(array)
            out_pixels += array.shape[0]
            array = torch.from_numpy(array)
            tmp_tensor = os.path.join(out, '{}.pth'.format(str(pth_count).rjust(7, '0')))
            torch.save(array, tmp_tensor)
            pth_count += 1

            for k, v in obj_cntr.items():
                if v.shape[0] > 5 * n_subset:
                    left_over[k] = v[:n_subset * 5, :]
                obj_cntr[k] = left_over[k]

        if (j + 1) % 100 == 0:
            print('{} items'.format(j + 1))

    print('\n{}'.format(mode.upper()))
    print('pixel count: {}, {} total'.format(pix_ct, out_pixels))

    if out_norm:
        pkl_name = os.path.join(out_norm, 'meanstd.pkl')
        with open(pkl_name, 'wb') as handle:
            pkl.dump((mean_, std_), handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    root = 'ta_data'

    data = '/nobackup/dketchu1/{}/cm'.format(root)
    if not os.path.isdir(data):
        data = '/media/nvm/{}/cm'.format(root)
    if not os.path.isdir(data):
        data = '/home/ubuntu/data/{}/cm'.format(root)

    images_dir = os.path.join(data, 'images')
    pixels_dir = os.path.join(data, 'pixels')

    for split in ['train', 'valid', 'test']:
        np_images = os.path.join(images_dir, split)
        pixel_dst = os.path.join(pixels_dir, split)
        if split == 'train':
            write_pixel_blocks(data_dir=np_images, out=pixel_dst, mode=split)
        else:
            write_pixel_blocks(data_dir=np_images, out=pixel_dst, mode=split)

# ========================= EOF ================================================================
