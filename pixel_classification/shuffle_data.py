import h5py
import numpy as np

def next_batch(file_map):
    '''File map: {class_code:{files:[], instances:int}}'''
    features = [] 
    labels = []
    for class_code in file_map:
        files = file_map[class_code]['files']
        n_instances = file_map[class_code]['instances']
        f = load_sample(n_instances, files)
        l = np.ones(f.shape[0])*class_code
        labels.append(l)
        features.append(f)
    feat_flat = [itm for sublist in features for itm in sublist]
    labels_flat = [itm for sublist in labels for itm in sublist]
    labels_flat = np.asarray(labels_flat)
    features_flat = np.asarray(feat_flat)
    return features_flat, labels_flat

def load_sample(required_instances, fnames):
    ''' Fnames: filenames of all files of class_code class
    required_instances: number of instances of training data required '''
    total_instances, num_files = get_total_instances(fnames)
    random_sample = np.random.choice(total_instances, required_instances, replace=False)
    random_sample.sort()
    ls = []
    last = 0
    offset = 0
    for f in fnames:
        with h5py.File(f, 'r') as hdf5:
            for key in hdf5:
                if hdf5[key].shape[0]:
                    last = offset
                    offset += hdf5[key].shape[0] 
                    indices = random_sample[random_sample < offset]
                    indices = indices[indices > last] 
                    try:
                        ls.append(hdf5[key][indices-last, :, :, :])
                    except UnboundLocalError as e:
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
