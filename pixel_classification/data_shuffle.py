import h5py
import numpy as np

def next_batch(file_map):
    '''File map: {class_code:{files:[], instances:int}}'''
    for class_code in file_map:
        files = file_map[class_code]['files']
        n_instances = file_map[class_code]['instances']




def load_sample(required_instances, fnames, class_code):
    ''' Fnames: filenames of all files of class_code class
    required_instances: number of instances of training data required '''
    total_instances, num_files = get_total_instances(fnames)
    random_sample = np.random.randint(0, total_instances, required_instances)
    random_sample.sort()
    ls = []
    last = 0
    offset = 0
    for f in fnames:
        with h5py.File(f, 'f') as hdf5:
            for key in hdf5:
                if hdf5[key].shape[0]:
                    frac_membership = int((hdf5[key].shape[0] / total_instances)*required_instances)
                    indices = sorted_sample[last:frac_membership] - sorted_sample[last] 
                    last = frac_membership
                    ls.append(hdf5[key][indices, :, :, :])
    ls = np.asarray(ls)
    return ls, np.ones((len(ls)))*class_code

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
