import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
from glob import glob
from random import sample, shuffle
from data_utils import generate_class_mask, get_shapefile_path_row
from rasterio import open as rasopen

NO_DATA = -1
MAX_POOLS = 3
CHUNK_SIZE = 1248 # some value that is evenly divisible by 2^3.
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
        # Need to save the dict object with a unique filename
        outfile = os.path.join(template, str(int(time.time())) + ".pkl")
        with open(outfile, 'wb') as f:
            pickle.dump(self.dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def set_data(self, data):
        self.dict['data'] = data

    def set_code(self, class_code):
        self.dict['class_code'] = class_code

    def set_class_mask(self, class_mask):
        self.dict['class_mask'] = class_mask


def create_training_data(target_dict, shapefile_directory, image_directory, training_directory):
    ''' target_dict: {filename or string in filename : class_code} '''
    done = set()
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
            master, meta = load_raster(master_raster)
            masks = []
            for match in all_matches:
                msk = generate_class_mask(match, mask_file)
                cc = assign_class_code(target_dict, match)
                if cc is not None:
                    dm = DataMask(msk, cc)
                    masks.append(dm)

            for i in range(0, master.shape[1], CHUNK_SIZE):
                for j in range(0, master.shape[2], CHUNK_SIZE):
                    sub_master = master[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                    for msk in masks:
                        s = msk.mask[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                        if not np.all(s == NO_DATA):
                            dt = DataTile(sub_master, s, msk.class_code)
                            dt.to_pickle(training_directory)

def all_matching_shapefiles(to_match, shapefile_directory):
    out = []
    pr = get_shapefile_path_row(to_match)
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if get_shapefile_path_row(f) == pr and to_match not in f:
            out.append(f)
    return out

def generate_balanced_data(shapefile_directory, image_directory, box_size, target, year=2013):
    train_raster = 'master_raster_' # templates
    mask_raster = 'class_mask_'
    ''' This is pretty much for binary classification.'''
    while True:
        for f in glob(os.path.join(shapefile_directory, "*.shp")):
            if target in f:
                all_matches = all_matching_shapefiles(f, shapefile_directory)
                p, r = get_shapefile_path_row(f)
                suffix = '{}_{}_{}.tif'.format(p, r, year)
                master_raster = os.path.join(image_directory, train_raster + suffix)
                mask_file = os.path.join(image_directory, mask_raster + suffix)
                if not os.path.isfile(master_raster):
                    print("Master raster not created for {}".format(suffix))
                    # TODO: More extensive error handling.
                else:
                    target_mask = generate_class_mask(f, mask_file)
                    class_mask = np.ones((NUM_CLASSES, target_mask.shape[1], target_mask.shape[2]))*NO_DATA
                    class_mask[1, :, :] = target_mask
                    required_instances = len(np.where(target_mask != NO_DATA)[0]) // (box_size*len(all_matches))
                    masks = []
                    for match in all_matches:
                        msk = generate_class_mask(match, mask_file)
                        #samp = random_sample(msk, required_instances, box_size)
                        #masks.append(samp)
                        masks.append(msk)

                    for i, s in enumerate(masks):
                        class_mask[0, :, :][s[0, :, :] != NO_DATA] = 1

                    master, meta = load_raster(master_raster)
                    for i in range(0, master.shape[1], CHUNK_SIZE):
                        for j in range(0, master.shape[2], CHUNK_SIZE):
                            sub_master = master[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                            sub_mask = class_mask[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                            if np.all(sub_mask == NO_DATA):
                                continue
                            else:
                                n_negative = len(np.where(sub_mask[0, :, :] != NO_DATA)[1]) 
                                positive = np.where(target_mask[:, :] != NO_DATA)
                                sorted_x = sorted(positive[1])
                                sorted_y = sorted(positive[2])
                                l = len(sorted_x) // 2
                                center_x = sorted_x[l]
                                center_y = sorted_y[l]
                                ofs = CHUNK_SIZE // 2
                                sub_positive = target_mask[:, center_x - ofs: center_x + ofs, center_y - ofs: center_y + ofs]
                                sub_master_positive = master[:, center_x - ofs: center_x + ofs, center_y - ofs: center_y + ofs]
                                required_instances = min(len(np.where(sub_positive[0, :, :] != NO_DATA)[1]), n_negative)
                                sub_negative = random_sample(sub_mask[0, :, :], required_instances,
                                        box_size=0, class_code=1)
                                sub_master_negative = sub_master
                                sub_positive = random_sample(sub_positive[0, :, :], required_instances,
                                       box_size=0, class_code=1)
                                one_hot_pos = np.ones((2, sub_positive.shape[0], sub_positive.shape[1]))*NO_DATA
                                one_hot_neg = np.ones((2, sub_negative.shape[0], sub_negative.shape[1]))*NO_DATA
                                one_hot_pos[1, :, :] = sub_positive
                                one_hot_neg[0, :, :] = sub_negative
                                sub_mas_pos, class_mask_pos = preprocess_data(sub_master_positive,
                                        one_hot_pos)
                                sub_mas_neg, class_mask_neg = preprocess_data(sub_master_negative,
                                        one_hot_neg)
                                ims = [sub_mas_pos, sub_mas_neg]
                                class_masks = [class_mask_pos, class_mask_neg]
                                for ii, jj in zip(ims, class_masks):
                                    yield ii, jj

class DataGen:

    def __init__(self, class_filename):
        self.file_list = None
        self.class_filename = class_filename
        self._get_files()
        self.n_files = len(self.file_list)
        self.first = True
        self.idx = 0
    
    def _get_files(self):
        self.file_list = [x[2] for x in os.walk(self.class_filename)][0]
        self.file_list = [os.path.join(self.class_filename, x) for x in self.file_list]

    def next(self):
        if self.idx == self.n_files:
            self.first = True
        if self.first:
            self.first = False
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


def generate_training_data(training_directory, box_size=0):
    ''' Assumes data is stored in training_directory
    in subdirectories labeled class_n_train
    and that n_classes is a global variable.'''
    class_dirs = [os.path.join(training_directory, x) for x in os.listdir(training_directory)]
    generators = []
    for d in class_dirs:
        generators.append(DataGen(d))
    # I want to 
    # a. shuffle the filenames to draw.
    # b. Define one epoch to be when we've iterated over all
    # examples of the class with the most training examples.
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
            samp = random_sample(subset['class_mask'], min_samples, box_size=box_size,
                    fill_value=1)
            one_hot = np.ones((NUM_CLASSES, samp.shape[1], samp.shape[2]))*NO_DATA
            one_hot[int(subset['class_code']), :, :] = samp
            for i in range(NUM_CLASSES):
                if i != int(subset['class_code']):
                    one_hot[i, :, :][samp[0, :, :] != NO_DATA] = 0
            subset['class_mask'] = one_hot

        masters = []
        masks = []
        for subset in data:
            master, mask = preprocess_data(subset['data'], subset['class_mask'])
            masters.append(master)
            masks.append(mask)

        outt = list(zip(masters, masks))
        shuffle(outt)
        for ms, msk in outt:
            msk = msk.astype(np.int32)
            yield ms, msk


def rotation(image, angle):
    return transform.rotate(image, angle, mode='constant', cval=NO_DATA)

def random_noise(image):
    return util.random_noise(image)

def h_flip(image):
    return image[:, ::-1]


def augment_data(image, class_mask):
    '''Randomly augments an image.'''
    if np.random.randint(2):
        deg = np.random.uniform(-25, 25)
        image = rotation(image, deg)
        class_mask = rotation(class_mask, deg)
    if np.random.randint(2):
        image = random_noise(image)
    if np.random.randint(2):
        image = h_flip(image)
        class_mask = h_flip(class_mask)
    if np.random.randint(2):
        image = np.flipud(image)
        class_mask = np.flipud(class_mask)
    return image, class_mask


def preprocess_data(master, mask, return_cuts=False):
    shp = master.shape
    rows = shp[1]; cols = shp[2]
    cut_rows = rows % (2**MAX_POOLS) 
    cut_cols = cols % (2**MAX_POOLS)
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
    training_directory = 'training_data'
    #create_training_data(target_dict, shapefile_directory, image_directory, training_directory)
    #generate_training_data(training_directory)
