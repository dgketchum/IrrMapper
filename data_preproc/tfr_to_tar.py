import os
import numpy as np
import tempfile
import shutil
import tarfile
import torch

from google.cloud import storage
from tf_dataset import make_test_dataset


def write_tfr_to_gcs(recs, bucket=None, bucket_dst=None, pattern='*gz', category='irrigated', start_count=0):
    """ Write tfrecord.gz to torch tensor, push .tar of torch tensor.pth to GCS bucket"""
    storage_client = storage.Client()

    def push_tar(t_dir, bckt, items, ind):
        tar_filename = '{}_{}.tar'.format(category, str(ind).zfill(6))
        tar_archive = os.path.join(t_dir, tar_filename)
        with tarfile.open(tar_archive, 'w') as tar:
            for i in items:
                tar.add(i, arcname=os.path.basename(i))
        bucket = storage_client.get_bucket(bckt)
        blob_name = os.path.join(bucket_dst, tar_filename)
        blob = bucket.blob(blob_name)
        print('push {}'.format(blob_name))
        blob.upload_from_filename(tar_archive)
        shutil.rmtree(t_dir)

    count = start_count

    dataset = make_test_dataset(recs, pattern).batch(1)
    obj_ct = np.array([0, 0, 0, 0])
    tmpdirname = tempfile.mkdtemp()
    items = []
    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy().squeeze()
        classes = np.array([np.any(labels[:, :, i]) for i in range(4)])
        obj_ct += classes
        features = features.numpy().squeeze()
        a = np.append(features, labels, axis=2)
        a = torch.from_numpy(a)
        tmp_name = os.path.join(tmpdirname, '{}.pth'.format(str(j).zfill(7)))
        torch.save(a, tmp_name)
        items.append(tmp_name)

        if len(items) == 20:
            push_tar(tmpdirname, bucket, items, count)
            tmpdirname = tempfile.mkdtemp()
            items = []
            count += 1

    if len(items) > 0:
        push_tar(tmpdirname, bucket, items, count)

    print(obj_ct)


def write_tfr_to_local(recs, out_dir, split, pattern='*gz', start_count=0):
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

    dataset = make_test_dataset(recs, pattern).batch(1)
    obj_ct = np.array([0, 0, 0])
    tmpdirname = tempfile.mkdtemp()
    items = []
    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy().squeeze()
        classes = np.array([np.any(labels[:, :, i]) for i in range(3)])
        obj_ct += classes
        features = features.numpy().squeeze()
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


if __name__ == '__main__':
    home = os.path.expanduser('~')
    # bucket_root = 'ta_data'
    # split = 'train'
    # bucket_dir = '/tar/{}'.format(split)
    # tf_recs = 'gs://ta_data/{}'.format(split)
    # glob_pattern = '*{}*gz'.format(split)
    # write_tfr_to_gcs(tf_recs, bucket=bucket_root, bucket_dst=bucket_dir, category=_type, start_count=0)

    for split in ['valid', 'train', 'test']:
        dir_ = '/media/hdisk/t_data/tfrecords/{}'.format(split)
        out_dir = '/media/hdisk/t_data/tarchives/images/{}'.format(split)
        write_tfr_to_local(dir_, out_dir, split=split, start_count=0)
# ========================= EOF ====================================================================