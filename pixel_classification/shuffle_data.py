import h5py
from collections import defaultdict
import numpy as np

def one_epoch(filenames, random_indices, class_code, chunk_size=5000):
    ''' Filename is the name of the data file,
        instances the number of instances that can fit in memory.
    '''
    if not isinstance(filenames, list):
        filenames = [filenames]
    for i in range(0, random_indices.shape[0], chunk_size):
        ret = load_sample(filenames, random_indices[i:i+chunk_size])
        yield ret, make_one_hot(np.ones((ret.shape[0]))*class_code, 4)

def make_one_hot(labels, n_classes):
    ret = np.zeros((len(labels), n_classes))
    for i, e in enumerate(labels):
        ret[i, int(e)] = 1
    return ret

def load_sample(fnames, random_indices):
    ''' Fnames: filenames of all files of class_code class
    required_instances: number of instances of training data required '''
    random_indices.sort()
    ls = []
    last = 0
    offset = 0
    for f in fnames:
        with h5py.File(f, 'r') as hdf5:
            for key in hdf5:
                if hdf5[key].shape[0]:
                    last = offset
                    offset += hdf5[key].shape[0] 
                    indices = random_indices[random_indices < offset]
                    indices = indices[indices > last] 
                    try:
                        ls.append(hdf5[key][indices-last, :, :, :])
                    except UnboundLocalError as e:
                        # When the index array is empty. This is
                        # an unhandled exception in the hdf5 library
                        pass

    flattened = [e for sublist in ls for e in sublist]
    return np.asarray(flattened)


def get_total_instances(fnames):
    total_instances = 0
    num_keys = 0
    for f in fnames:
        with h5py.File(f, 'r') as hdf5:
            for key in hdf5:
                if hdf5[key].shape[0]:
                    total_instances += hdf5[key].shape[0]
                    num_keys += 1
    return total_instances, num_keys

if __name__ == '__main__':
    pass
