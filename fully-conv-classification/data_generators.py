import numpy as np
import os
import time
import pickle
import sys
import matplotlib.pyplot as plt
from glob import glob
from random import sample, shuffle
from sklearn.utils.class_weight import compute_class_weight
from data_utils import generate_class_mask, get_shapefile_path_row
from rasterio import open as rasopen
from warnings import warn
from skimage import transform

NO_DATA = -1
CHUNK_SIZE = 608 # some value that is evenly divisible by 2^MAX_POOLS.
NUM_CLASSES = 4

def random_sample(class_mask, n_instances, box_size=0, fill_value=1):
    if box_size:
        n_instances /= box_size

    out = np.where(class_mask != NO_DATA)
    class_mask = class_mask.copy()
    try:
        out_x = out[1]
        out_y = out[2] 
    except IndexError as e:
        out_x = out[0]
        out_y = out[1] 

    indices = np.random.choice(len(out_x), size=n_instances, replace=False)
    out_x = out_x[indices]
    out_y = out_y[indices] 

    try:
        class_mask[:, :, :] = NO_DATA
        if box_size == 0:
            class_mask[0, out_x, out_y] = fill_value
        else:
            ofs = box_size // 2
            for x, y in zip(out_x, out_y):
                class_mask[0, x-ofs:x+ofs+1, y-ofs:y+ofs+1] = fill_value

    except IndexError as e:
        class_mask[:, :] = NO_DATA
        if box_size == 0:
            class_mask[out_x, out_y] = fill_value
        else:
            ofs = box_size // 2
            for x, y in zip(out_x, out_y):
                class_mask[x-ofs:x+ofs, y-ofs:y+ofs] = fill_value

    return class_mask

def load_raster(master_raster):
    with rasopen(master_raster, 'r') as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta


def assign_class_code(target_dict, shapefilename):
    for key in target_dict:
        if key in shapefilename:
            return target_dict[key]
    print("{} has no known match in target_dict.".format(shapefilename))
    return None


class DataMask(object):

    def __init__(self, mask, class_code):
        self.mask = mask
        self.class_code = class_code


class DataTile(object):

    def __init__(self, data, class_mask, class_code):
        self.dict = {}
        self.dict['data'] = data
        self.dict['class_mask'] = class_mask
        self.dict['class_code'] = class_code

    def to_pickle(self, training_directory):
        if not os.path.isdir(training_directory):
            os.mkdir(training_directory)
        template = os.path.join(training_directory,
                'class_{}_data/'.format(self.dict['class_code']))
        if not os.path.isdir(template):
            os.mkdir(template)
        outfile = os.path.join(template, str(time.time()) + ".pkl")
        if not os.path.isfile(outfile):
            with open(outfile, 'wb') as f:
                pickle.dump(self.dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("What? Contact administrator.")

    def set_data(self, data):
        self.dict['data'] = data

    def set_code(self, class_code):
        self.dict['class_code'] = class_code

    def set_class_mask(self, class_mask):
        self.dict['class_mask'] = class_mask


def create_training_data(target_dict, shapefile_directory, image_directory, training_directory,
        count, save=True):
    ''' target_dict: {filename or string in filename : class_code} '''
    done = set()
    pixel_dict = {} # counts number of pixels present in each class.
    for class_code in target_dict.values():
        pixel_dict[class_code] = 0
    year = 2013
    train_raster = 'master_raster_'
    mask_raster = 'class_mask_'
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if f not in done:
            all_matches = all_matching_shapefiles(f, shapefile_directory)
            done.add(f)
            for match in all_matches:
                done.add(match)
            p, r = get_shapefile_path_row(f)
            suffix = '{}_{}_{}.tif'.format(p, r, year)
            master_raster = os.path.join(image_directory, train_raster + suffix)
            mask_file = os.path.join(image_directory, mask_raster + suffix) # for rasterio.mask.mask
            # this file is projected the same as the shapefile.
            masks = []
            all_matches.append(f)
            shp = None
            for match in all_matches:
                msk = generate_class_mask(match, mask_file)
                shp = msk.shape
                cc = assign_class_code(target_dict, match)
                if cc is not None:
                    dm = DataMask(msk, cc)
                    masks.append(dm)
            if save:
                master, meta = load_raster(master_raster)
            else:
                master = np.zeros(shp)
            
            for i in range(0, master.shape[1], CHUNK_SIZE):
                for j in range(0, master.shape[2], CHUNK_SIZE):
                    sub_master = master[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                    for msk in masks:
                        s = msk.mask[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                        if not np.all(s == NO_DATA):
                            pixel_dict[msk.class_code] += len(np.where(s != NO_DATA)[0])
                            count += 1
                            if save:
                                dt = DataTile(sub_master, s, msk.class_code)
                                dt.to_pickle(training_directory)
    return count, pixel_dict


def all_matching_shapefiles(to_match, shapefile_directory):
    out = []
    pr = get_shapefile_path_row(to_match)
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if get_shapefile_path_row(f) == pr and to_match not in f:
            out.append(f)
    return out


class DataGen:

    def __init__(self, class_filename):
        self.file_list = None
        self.class_filename = class_filename
        self._get_files()
        self.n_files = len(self.file_list)
        self.idx = 0
    
    def _get_files(self):
        self.file_list = [x[2] for x in os.walk(self.class_filename)][0]
        self.file_list = [os.path.join(self.class_filename, x) for x in self.file_list]

    def next(self):
        if self.idx == self.n_files or self.idx == 0:
            self.idx = 0
            self.shuffled = sample(self.file_list, self.n_files)
            out = self.shuffled[self.idx]
            self.idx += 1
        else:
            out = self.shuffled[self.idx]
            self.idx += 1
        return self._from_pickle(out)
    
    def _from_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data


def generate_training_data(training_directory, max_pools, random_sample=True, train=True, box_size=0):
    ''' Assumes data is stored in training_directory
    in subdirectories labeled class_n_train
    and that n_classes is a global variable.'''
    class_dirs = [os.path.join(training_directory, x) for x in os.listdir(training_directory)]
    generators = []
    for d in class_dirs:
        generators.append(DataGen(d))
    # TODO: Apply image augmentation.
    while True:
        min_samples = np.inf
        data = []
        for gen in generators:
            out = gen.next().copy()
            data.append(out)
            n_samples = len(np.where(out['class_mask'] != NO_DATA)[0])
            if n_samples < min_samples:
                min_samples = n_samples
        for subset in data:
            if random_sample:
                samp = random_sample(subset['class_mask'], min_samples, box_size=box_size,
                        fill_value=1)
            else:
                samp = subset['class_mask']
                samp[samp != NO_DATA] = 1
            one_hot = np.ones((NUM_CLASSES, samp.shape[1], samp.shape[2]))*NO_DATA
            one_hot[int(subset['class_code']), :, :] = samp
            for i in range(NUM_CLASSES):
                if i != int(subset['class_code']):
                    one_hot[i, :, :][samp[0, :, :] != NO_DATA] = 0
            subset['class_mask'] = one_hot

        masters = []
        masks = []
        for subset in data:
            master, mask = preprocess_data(subset['data'], subset['class_mask'], max_pools)
            masters.append(master)
            masks.append(mask)

        if train:
            augmented_masters = []
            augmented_masks = []
            for master, mask in zip(masters, masks):
                ms, msk = augment_data(master, mask)
                augmented_masters.append(ms)
                augmented_masks.append(msk)

            masters += augmented_masters
            masks += augmented_masks

        # Shuffle order of data here?
        for ms, msk in zip(masters, masks):
            yield ms, msk


def rotation(image, angle):
    return transform.rotate(image, angle, mode='constant', cval=NO_DATA)

def h_flip(image):
    return image[:, ::-1]

def augment_data(image, class_mask):
    '''Randomly augments an image.'''
    # if np.random.randint(2):
    #     deg = np.random.uniform(-25, 25)
    #     image = rotation(image, deg)
    #     class_mask = rotation(class_mask, deg)
    if np.random.randint(2):
        image = h_flip(image)
        class_mask = h_flip(class_mask)
    if np.random.randint(2):
        image = np.flipud(image)
        class_mask = np.flipud(class_mask)
    return image, class_mask


def preprocess_data(master, mask, max_pools, return_cuts=False):
    ''' This function preprocesses data in such a way
    that downscaling it by 2 max_pools times will result 
    in an input that plays nicely with the FCNN expecting it.
    Master, mask in this example are tiles from the original image.'''
    shp = master.shape
    rows = shp[1]; cols = shp[2]
    if max_pools != 0:
        cut_rows = rows % (2**max_pools) 
        cut_cols = cols % (2**max_pools)
    else:
        cut_rows = 0
        cut_cols = 0
    out_m = np.zeros((1, shp[0], shp[1] - cut_rows, shp[2] - cut_cols))

    if cut_cols != 0 and cut_rows != 0:
        out_m[0, :, :, :] = master[:, :-cut_rows, :-cut_cols]
        shp = mask.shape
        out_mask = np.zeros((1, shp[0], shp[1] - cut_rows, shp[2] - cut_cols))
        out_mask[0, :, :, :] = mask[:, :-cut_rows, :-cut_cols]
    elif cut_cols == 0 and cut_rows != 0:
        out_m[0, :, :, :] = master[:, :-cut_rows, :]
        shp = mask.shape
        out_mask = np.zeros((1, shp[0], shp[1] - cut_rows, shp[2] - cut_cols))
        out_mask[0, :, :, :] = mask[:, :-cut_rows, :]
    elif cut_cols != 0 and cut_rows == 0:
        out_m[0, :, :, :] = master[:, :, :-cut_cols]
        shp = mask.shape
        out_mask = np.zeros((1, shp[0], shp[1] - cut_rows, shp[2] - cut_cols))
        out_mask[0, :, :, :] = mask[:, :, :-cut_cols]
    else:
        out_m[0, :, :, :] = master[:, :, :]
        shp = mask.shape
        out_mask = np.zeros((1, shp[0], shp[1] - cut_rows, shp[2] - cut_cols))
        out_mask[0, :, :, :] = mask[:, :, :]

    out_m = np.swapaxes(out_m, 1, 3)
    out_mask = np.swapaxes(out_mask, 1, 3)
    if return_cuts:
        return out_m, out_mask, cut_rows, cut_cols

    return out_m, out_mask


if __name__ == '__main__':
    shapefile_directory = 'shapefile_data/'
    image_directory = 'master_rasters/'
    irr1 = 'Huntley'
    irr2 = 'Sun_River'
    fallow = 'Fallow'
    forest = 'Forrest'
    other = 'other'
    target_dict = {irr2:0, irr1:0, fallow:1, forest:2, other:3}
    year = 2013
    train_dir = 'training_data/multiclass/train/'
    shp_train = 'shapefile_data/train/'
    count = 0
    save = False
    count, pixel_dict = create_training_data(target_dict, shp_train, image_directory, train_dir,
            count, save=save)
    print("You have {} instances per training epoch.".format(count))
    print("And {} instances in each class.".format(pixel_dict))

    max_weight = max(pixel_dict.values())
    for key in pixel_dict:
        print(key, max_weight / pixel_dict[key])
    tot = 0
    test_dir = 'training_data/multiclass/test/'
    shp_test = 'shapefile_data/test/'
    count = 0
    count, pixel_dict = create_training_data(target_dict, shp_test, image_directory, test_dir, 
            count, save=save)
    print("You have {} instances per test epoch.".format(count))
    print("And {} instances in each class.".format(pixel_dict))
