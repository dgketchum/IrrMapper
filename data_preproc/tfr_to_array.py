import os
import numpy as np
import tempfile
import shutil
import tarfile
import torch

from tfr_utils import make_test_dataset
from data_preproc.plot_data import plot_image_data

N_CLASSES = 3
PLOTS = '/media/hdisk/t_data/tfrecords/plots'


def write_tfr_to_local(recs, out_dir, split, pattern='*gz', start_count=0, plot=False):
    """ Write tfrecord.gz to torch tensor, push .tar of torch tensor.pth to local"""

    def push_tar(t_dir, out_dir, mode, items, ind, prefix=None):
        if prefix:
            tar_filename = '{}_{}_{}.tar'.format(prefix, mode, str(ind).zfill(6))
        else:
            tar_filename = '{}_{}.tar'.format(mode, str(ind).zfill(6))
        tar_archive = os.path.join(out_dir, tar_filename)
        with tarfile.open(tar_archive, 'w') as tar:
            for i in items:
                tar.add(i, arcname=os.path.basename(i))
        shutil.rmtree(t_dir)

    count = start_count

    dataset = make_test_dataset(recs, pattern)
    obj_ct = np.zeros((1, N_CLASSES))
    tmpdirname = tempfile.mkdtemp()
    items = []
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
        tmp_name = os.path.join(tmpdirname, '{}.pth'.format(str(j).zfill(7)))
        torch.save(a, tmp_name)
        items.append(tmp_name)

        if len(items) == 20:
            push_tar(tmpdirname, out_dir, split, items, count)
            tmpdirname = tempfile.mkdtemp()
            items = []
            count += 1

    if len(items) > 0:
        push_tar(tmpdirname, out_dir, split, items, count)

    print(split, obj_ct)
    print('{} shards'.format(j + 1))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    # bucket_root = 'ta_data'
    # split = 'train'
    # bucket_dir = '/tar/{}'.format(split)
    # tf_recs = 'gs://ta_data/{}'.format(split)
    # glob_pattern = '*{}*gz'.format(split)
    # write_tfr_to_gcs(tf_recs, bucket=bucket_root, bucket_dst=bucket_dir, category=_type, start_count=0)

    for split in ['train', 'test', 'valid']:
        dir_ = '/nobackup/dketchu1/ts_data/tfrecords/{}'.format(split)
        out_dir = '/nobackup/dketchu1/ts_data/images/{}'.format(split)
        write_tfr_to_local(dir_, out_dir, split=split, start_count=0, plot=False)
# ========================= EOF ====================================================================
