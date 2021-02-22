import os

from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/dgketchum/ssebop-montana-57d2b4da4339.json'


def get_bucket_contents(bucket_name, glob=None):
    client = storage.Client()
    dct = {}
    empty = []
    for blob in client.list_blobs(bucket_name, prefix=glob):
        dirname = os.path.dirname(blob.name)
        b_name = os.path.basename(blob.name)
        if blob.size == 20:
            empty.append(blob.name)
            continue
        size = blob.size / 1e6
        if dirname not in dct.keys():
            dct[dirname] = [(b_name, size)]
        else:
            dct[dirname].append((b_name, size))
    return dct, empty


def move_empty(bucket_name, glob=None, dst_folder=None):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=glob):
        dirname = os.path.dirname(blob.name)
        if blob.size == 20:
            if 'unproc' in dirname:
                out_name = os.path.join(dst_folder, os.path.basename(blob.name))
                print('move {} to {}'.format(blob.name, out_name))
                bucket.rename_blob(blob, out_name)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
