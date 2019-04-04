import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
from glob import glob
from random import sample, shuffle
from skimage.segmentation import find_boundaries
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
from runspec import mask_rasters
from data_utils import load_raster
from shapefile_utils import get_shapefile_path_row, generate_class_mask 
from rasterio import open as rasopen
from warnings import warn
from skimage import transform
from sat_image.warped_vrt import warp_single_image

NO_DATA = -1
NUM_CLASSES = 4


def weight_map(mask, w0=10, sigma=10):
    mask = mask.copy().astype(bool)
    mask = ~mask # make the non-masked areas masked
    distances = distance_transform_edt(mask) # ask where the closest masked pixel is
    # distances are always positive, so 1-distances can be very negative.
    # We're setting the e-folding time with sigma, and the
    # border pixel value (y-intercept) with w0.
    return w0*np.exp((1-distances) / sigma)


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


def get_masks(image_directory):
    ''' Returns all raster masks in the image directory.'''
    paths = []
    for dirpath, dirnames, filenames in os.walk(image_directory):
        for f in filenames:
            for suffix in mask_rasters():
                if f.endswith(suffix):
                    paths.append(os.path.join(dirpath, f))
    out = None
    first_geo = None
    n_masks = len(paths)
    first = True
    for mask_file in paths:
        mask, meta = load_raster(mask_file)
        # mask value here is 1.
        if first:
            first = False
            first_geo = meta.copy()
            out = np.zeros((mask.shape[1], mask.shape[2]))
        try:
            out[mask[0] == 1] = 1 # 0 index is for removing the (1, n, m) dimension.
        except (ValueError, IndexError) as e:
            mask = warp_single_image(mask_file, first_geo)
            out[mask[0] == 1] = 1
    return out


def extract_training_data_unet(target_dict, shapefile_directory, image_directory,
        master_raster_directory, training_directory, count, save=True, chunk_size=572):
    ''' target_dict: {filename or string in filename : class_code} '''
    unet_output_size = 388 # input to unet.
    unet_input_size = 572 # input to unet.
    
    # this means the class mask must be 388x388 - chop 184 from height/width.
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
            if "37_27" in suffix:
                continue
            fmask = get_masks(os.path.join(image_directory, suffix[:-4]))
            master_raster = os.path.join(master_raster_directory, train_raster + suffix)
            mask_file = os.path.join(master_raster_directory, mask_raster + suffix) # for rasterio.mask.mask
            masks = [] 
            all_matches.append(f)
            shp = None
            for match in all_matches:
                msk = generate_class_mask(match, mask_file)
                # try:
                #     msk[0][fmask == 1] = NO_DATA
                # except IndexError:
                #     print(match, msk.shape, fmask.shape)
                #     # What's going on here?
                #     # Fmasks and masks have different shapes...
                #     # Probably need to warp_vrt?
                #     #msk[:, :][fmask == 1] = NO_DATA

                shp = msk.shape
                cc = assign_class_code(target_dict, match)
                if cc is not None:
                    dm = DataMask(msk, cc)
                    masks.append(dm)
            if save:
                master, meta = load_raster(master_raster)
            else:
                master = np.zeros(shp)
            
            # 92 is unet offset.
            for i in range(92, master.shape[1], unet_output_size):
                for j in range(92, master.shape[2], unet_output_size):
                    sub_master = master[:, i-92:i+unet_output_size+92, j-92:j+unet_output_size+92]
                    if sub_master.shape[1] == unet_input_size and sub_master.shape[2] == unet_input_size:
                        for msk in masks:
                            s = msk.mask[:, i:i+unet_output_size, j:j+unet_output_size]
                            if not np.all(s == NO_DATA):
                                pixel_dict[msk.class_code] += len(np.where(s != NO_DATA)[0])
                                count += 1
                                if save:
                                    dt = DataTile(sub_master, s, msk.class_code)
                                    dt.to_pickle(training_directory)
    return count, pixel_dict


def extract_training_data(target_dict, shapefile_directory, image_directory,
        master_raster_directory, training_directory, count, save=True, chunk_size=572):
    ''' target_dict: {filename or string in filename : class_code} '''
    done = set()
    pixel_dict = {} # counts number of pixels present in each class.
    for class_code in target_dict.values():
        pixel_dict[class_code] = 0
    year = 2013
    train_raster = 'master_raster_'
    mask_raster = 'class_mask_'
    CHUNK_SIZE = chunk_size
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if f not in done:
            all_matches = all_matching_shapefiles(f, shapefile_directory)
            done.add(f)
            for match in all_matches:
                done.add(match)
            p, r = get_shapefile_path_row(f)
            suffix = '{}_{}_{}.tif'.format(p, r, year)
            fmask = get_masks(os.path.join(image_directory, suffix[:-4]))
            master_raster = os.path.join(master_raster_directory, train_raster + suffix)
            mask_file = os.path.join(master_raster_directory, mask_raster + suffix) # for rasterio.mask.mask
            masks = [] # these are class masks for the labelling of data.
            all_matches.append(f)
            shp = None
            for match in all_matches:
                msk = generate_class_mask(match, mask_file)
                msk[0][fmask == 1] = NO_DATA
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
                    if sub_master.shape[1] == CHUNK_SIZE and sub_master.shape[2] == CHUNK_SIZE:
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


def generate_training_data(training_directory, max_pools, sample_random=True, box_size=0, 
        batch_size=8, class_weights={}, threshold=0.9, w0=40, sigma=10, channels='all', 
        train=True):
    ''' Assumes data is stored in training_directory
    in subdirectories labeled class_n_train with n the class code '''
    class_dirs = [os.path.join(training_directory, x) for x in os.listdir(training_directory)]
    generators = []
    border_class = len(class_weights.keys())
    for d in class_dirs:
        generators.append(DataGen(d))
    while True:
        masters = []
        masks = []
        weightings = []
        for _ in range(2):
            min_samples = np.inf
            data = []
            for gen in generators:
                out = gen.next().copy()
                data.append(out)
                if sample_random:
                    n_samples = len(np.where(out['class_mask'] != NO_DATA)[0])
                    if n_samples < min_samples:
                        min_samples = n_samples

            first = False
            one_hot = None
            for subset in data:
                if sample_random:
                    samp = random_sample(subset['class_mask'], min_samples, box_size=box_size,
                            fill_value=subset['class_code'])
                else:
                    samp = subset['class_mask']
                    samp[samp != NO_DATA] = subset['class_code']

                subset['class_mask'] = samp

            for subset in data:
                master, mask = preprocess_data(subset['data'], subset['class_mask'], max_pools)
                if channels == 'all':
                    master = np.squeeze(master)
                else:
                    master = master[:, :, :, channels]
                    master = np.squeeze(master)
                mask = mask[0, :, :, 0] 
                mask[mask != -1] = 1 # make the mask binary.
                mask[mask == -1] = 0 # -1 is NO_DATA.
                weights = weight_map(mask, w0=w0, sigma=sigma) # create weight map
                labels = weights.copy()
                labels[labels >= threshold] = border_class 
                labels[mask == 1] = subset['class_code']
                weights[weights < threshold] = 0 # threshold the weight values arbitrarily
                weights[weights != 0] = 0 #remove the border weights
                weights[mask == 1] = class_weights[subset['class_code']] 
                multidim_weights = np.zeros((weights.shape[0], weights.shape[1], border_class)) #
                one_hot = np.zeros((labels.shape[0], labels.shape[1], border_class))
                #one_hot[:, :, border_class][labels == border_class] = 1
                one_hot[:, :, subset['class_code']][labels == subset['class_code']] = 1
                # above is circular but will allow for changing to a sparse encoding easily
                for i in range(border_class):
                    multidim_weights[:, :, i] = weights
                if not train:
                    multidim_weights[multidim_weights != 0] = 1 
                masters.append(master)
                masks.append(one_hot)
                weightings.append(multidim_weights)

        yield [np.asarray(masters, dtype=np.float32), np.asarray(weightings)], np.asarray(masks)





def rotation(image, angle):
    return transform.rotate(image, angle, mode='constant', cval=NO_DATA)


def h_flip(image):
    return image[:, ::-1]


def augment_data(image, class_mask):
    '''Randomly augments an image.'''
    if np.random.randint(2):
        image = h_flip(image)
        class_mask = h_flip(class_mask)
    if np.random.randint(2):
        image = np.flipud(image)
        class_mask = np.flipud(class_mask)
    return image, class_mask


def preprocess_data(master, mask, max_pools, return_cuts=False):
    ''' This function preprocesses data in such a way
    so it will work with a FCNN with an arbitrary number of max pools.
    Master, mask in this function are tiles from the original image.'''
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

    master_train = 'master_rasters/train/'
    master_test = 'master_rasters/test/'
    image_train = 'image_data/train/' # for fmasks.
    image_test = 'image_data/test/' # for fmasks.
    irr1 = 'Huntley'
    irr2 = 'Sun_River'
    fallow = 'Fallow'
    forest = 'Forrest'
    other = 'other'
    target_dict = {irr2:0, irr1:0, fallow:1, forest:2, other:3}
    train_dir = 'training_data/multiclass/train/'
    shp_train = 'shapefile_data/train/'
    count = 0
    save = True
    count, pixel_dict = extract_training_data_unet(target_dict, shp_train, image_train,
            master_train, train_dir, count, save=save) 
    # Need to parallelize the extraction of training data.
    # Or maybe not. It seems like parallelizing the opening/closing
    # of rasters can stomp on the data.
    print("You have {} instances per training epoch.".format(count))
    print("And {} instances in each class.".format(pixel_dict))
    max_weight = max(pixel_dict.values())
    for key in pixel_dict:
        print(key, max_weight / pixel_dict[key])
    tot = 0
    test_dir = 'training_data/multiclass/test/'
    shp_test = 'shapefile_data/test/'
    count = 0
    count, pixel_dict = extract_training_data_unet(target_dict, shp_test, image_test, master_test, 
            test_dir, count, save=save)
    print("You have {} instances per test epoch.".format(count))
    print("And {} instances in each class.".format(pixel_dict))
