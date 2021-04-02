import os
import numpy as np
import torch

from tfr_utils import make_test_dataset
from data_preproc.plot_data import plot_image_data

N_CLASSES = 3
PLOTS = '/media/hdisk/t_data/tfrecords/plots'


def write_tfr_to_local(recs, out_dir, split, plot=False):
    """ Write tfrecord.gz to torch tensor, push torch tensor.pth to local"""

    dataset = make_test_dataset(recs, '*gz')
    obj_ct = np.zeros((1, N_CLASSES))

    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy()
        # impervious was cast to 4, force to 3
        labels[labels > 3] = 3
        classes = np.array([np.any(labels == i) for i in range(N_CLASSES)])
        obj_ct += classes
        features = features.numpy().squeeze()

        if plot:
            out_plot = os.path.join(PLOTS, '{}.png'.format(j))
            plot_image_data(features, labels, out_file=out_plot)

        labels = labels.reshape(labels.shape[0], labels.shape[1], 1)
        a = np.append(features, labels, axis=2)
        a = torch.from_numpy(a)
        tmp_name = os.path.join(out_dir, '{}.pth'.format(str(j).zfill(7)))
        torch.save(a, tmp_name)
        if (j + 1) % 100 == 0:
            print('{} items'.format(j + 1))

    print(split, obj_ct)
    print('{} shards'.format(j + 1))


if __name__ == '__main__':

    data = '/nobackup/dketchu1/ts_data/nm'

    if not os.path.isdir(data):
        data = '/media/nvm/ts_data/nm'

    for split in ['valid', 'test', 'train']:
        dir_ = os.path.join(data, 'tfrecords', '{}'.format(split))
        out_dir = os.path.join(data, 'images', '{}'.format(split))
        write_tfr_to_local(dir_, out_dir, split=split, plot=False)
# ========================= EOF ====================================================================
