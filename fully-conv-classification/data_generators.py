import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
from glob import glob, iglob
from random import sample, shuffle, choice
from scipy.ndimage.morphology import distance_transform_edt
from runspec import mask_rasters
from data_utils import load_raster
from shapefile_utils import get_shapefile_path_row, generate_class_mask 
from rasterio import open as rasopen
from warnings import warn
from skimage import transform
from sat_image.warped_vrt import warp_single_image
from tensorflow.keras.utils import Sequence


def distance_map(mask):
    mask = mask.copy().astype(bool)
    mask = ~mask # make the non-masked areas masked
    distances = distance_transform_edt(mask) # ask where the closest masked pixel is
    # distances are always positive, so 1-distances can be very negative.
    # We're setting the e-folding time with sigma, and the
    # border pixel value (y-intercept) with w0.
    return distances


def weight_map(mask, w0=10, sigma=10):
    mask = mask.copy().astype(bool)
    mask = ~mask # make the non-masked areas masked
    distances = distance_transform_edt(mask) # ask where the closest masked pixel is
    # distances are always positive, so 1-distances can be very negative.
    # We're setting the e-folding time with sigma, and the
    # border pixel value (y-intercept) with w0.
    return w0*np.exp((1-distances) / sigma)


def random_sample(class_mask, n_instances, box_size=0, fill_value=1, nodata=0):
    if box_size:
        n_instances /= box_size

    out = np.where(class_mask != nodata)
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
        class_mask[:, :, :] = nodata
        if box_size == 0:
            class_mask[0, out_x, out_y] = fill_value
        else:
            ofs = box_size // 2
            for x, y in zip(out_x, out_y):
                class_mask[0, x-ofs:x+ofs+1, y-ofs:y+ofs+1] = fill_value

    except IndexError as e:
        class_mask[:, :] = nodata
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


def concatenate_fmasks(image_directory, class_mask, class_mask_geo, nodata=0):
    paths = []
    for dirpath, dirnames, filenames in os.walk(image_directory):
        for f in filenames:
            for suffix in mask_rasters():
                if f.endswith(suffix):
                    paths.append(os.path.join(dirpath, f))
    for fmask_file in paths:
        fmask, fmeta = load_raster(fmask_file)
        try:
            class_mask[fmask == 1] = nodata # 0 index is for removing the (1, n, m) dimension.
        except (ValueError, IndexError) as e:
            fmask = warp_single_image(fmask_file, class_mask_geo)
            class_mask[fmask == 1] = nodata
    
    return class_mask


def extract_training_data(target_dict, shapefile_directory, image_directory,
        master_raster_directory, training_directory, save=True, tile_size=608,
        year=2013, min_pixels=2000, fmask=True, nodata=0, augment_dict={}):
    '''
    target_dict: {filename or string in filename : class_code}
    This function extracts training data from master_rasters in master_raster_directory. Master
    rasters are rasters containing all feature bands. 
    To do this, we iterate over the shapefile (vector) data in shapefile directory
    and mask the corresponding raster with the vector data. We also ignore all pixels
    in the master_rasters that have any clouds or water in them. 
    steps:
       pull a shapefile at random from shapefile_directory.
       get all the other shapefiles that are in the same path and row.
       use these shapefiles to create a binary mask: 0 where there's no
       data and 1 where there is data. 
       Assign each binary mask to a class.
       Iterate over the master raster in that path / row and
       create a new datatile object for each binary mask that contains
       data, and save it. 
    '''
    
    done = set()
    pixel_dict = {} # pixel dict counts number of pixels present in each class.
    for class_code in target_dict.values():
        pixel_dict[class_code] = 0 
    year = year #TODO: incorporate year attr. from shapefile
    train_raster = 'master_raster_' # template
    mask_raster = 'class_mask_' # template
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if f not in done:
            done.add(f)
            all_matches = all_matching_shapefiles(f, shapefile_directory) # get all shapefiles in the same path / row
            for match in all_matches:
                done.add(match)
            p, r = get_shapefile_path_row(f)
            suffix = '{}_{}_{}.tif'.format(p, r, year) #image directory
            master_raster = os.path.join(master_raster_directory, train_raster + suffix)
            mask_file = os.path.join(master_raster_directory, mask_raster + suffix) # for rasterio.mask.mask
            masks = [] 
            shp = None
            for match in all_matches:
                cc = assign_class_code(target_dict, match)
                msk, mask_meta = generate_class_mask(match, mask_file, nodata=nodata)
                if fmask:
                    msk = concatenate_fmasks(os.path.join(image_directory, suffix[:-4]), msk,
                            mask_meta, nodata=nodata) # Need to make sure this is doing what I expect.
                shp = msk.shape
                print(match, cc)
                if cc is not None:
                    dm = DataMask(msk, cc) # a binary mask that has a class_code attributed to it.
                    masks.append(dm)

            if save:
                master, meta = load_raster(master_raster)
            else:
                master = np.zeros(shp)

            for datamask in masks:
                if augment_dict[datamask.class_code]:
                    pixel_dict = _iterate_over_raster(master, datamask, pixel_dict, 
                            tile_size=tile_size, save=save, augment=True, min_pixels=min_pixels,
                            training_directory=training_directory)
                else:
                    pixel_dict = _iterate_over_raster(master, datamask,
                            pixel_dict, min_pixels=min_pixels,
                            tile_size=tile_size, save=save,
                            training_directory=training_directory)

    return pixel_dict


def _iterate_over_raster(raster, datamask, pixel_dict, tile_size=608, augment=False,
        save=True, training_directory=None, min_pixels=None):
    step = tile_size
    if augment:
        step = np.random.randint(50, tile_size // 2)
        print("Augmenting w/ step:", step)
    for i in range(0, raster.shape[1]-tile_size, step):
        for j in range(0, raster.shape[2]-tile_size, step):
            sub_raster = raster[:, i:i+tile_size, j:j+tile_size]
            sub_mask = datamask.mask[:, i:i+tile_size, j:j+tile_size]
            if _check_dimensions_and_content(sub_raster, sub_mask, tile_size, min_pixels):
                pixel_dict[datamask.class_code] += len(np.where(sub_mask != 0)[0])
                if save:
                    dt = DataTile(sub_raster, sub_mask, datamask.class_code)
                    dt.to_pickle(training_directory)
    return pixel_dict


def _check_dimensions_and_content(sub_raster, sub_mask, tile_size, min_pixels):
    if len(np.where(sub_mask != 0)[0]) < min_pixels:
        return False
    if sub_mask.shape[1] != tile_size or sub_mask.shape[2] != tile_size:
        return False
    if sub_raster.shape[1] != tile_size or sub_raster.shape[2] != tile_size:
        return False
    return True


def all_matching_shapefiles(to_match, shapefile_directory):
    out = []
    pr = get_shapefile_path_row(to_match)
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if get_shapefile_path_row(f) == pr:
            out.append(f)
    return out


class DataGen:
    ''' Infinite data generator. Pulls files from 
        a directory named "class_dir".
        Class dir can have multiple directories full of data files
        in it.
    '''

    def __init__(self, class_dir, augment=False, random_augment=False):
        self.file_list = None
        self.class_dir = class_dir
        self._get_files()
        self.n_files = len(self.file_list)
        self.idx = 0
        self.shuffled = sample(self.file_list, self.n_files)

    def _get_files(self):
        self.file_list = [x for x in iglob(self.class_dir + "**", recursive=True)]
        self.file_list = [x for x in self.file_list if os.path.isfile(x)]

    def next(self):
        if self.idx == self.n_files:
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


def make_border_labels(mask, border_width):
    ''' Border width: Pixel width. '''
    dm = distance_map(mask)
    dm[dm > border_width] = 0
    return dm


class SatDataSequence(Sequence):

    def __init__(self, data_directory, batch_size, class_weights={},
            border_width=1, classes_to_augment=None):
        self.data_directory = data_directory
        self.class_weights = class_weights
        self.batch_size = batch_size
        self.classes_to_augment = classes_to_augment
        self.border_width = border_width
        self._get_files()
        self.n_files = len(self.file_list)
        self.idx = 0
        self.shuffled = sample(self.file_list, self.n_files)


    def _get_files(self):
        # Now, get n lists where n is the number of classes (excluding border class).
        # Then, sample from the minority lists until we have 
        # the same number of data tiles from each class, then concatenate
        # all the lists and shuffle. on epoch end, do this process again.
        self.file_dict = {}
        i = 0
        for (dirpath, dirnames, filenames) in os.walk(self.data_directory):
            if dirpath != self.data_directory:
                self.file_dict[i] = [os.path.join(dirpath, x) for x in filenames]
                i += 1
        self.lengths = [len(self.file_dict[k]) for k in self.file_dict]
        self._create_file_list()


    def _create_file_list(self):
        max_instances = max(self.lengths)
        self.file_list = []
        for class_dir in self.file_dict:
            files = self.file_dict[class_dir]
            self.file_list.extend(files)
            if len(files) != max_instances:
                if len(files) < (max_instances - len(files)):
                    files *= (max_instances // len(files))
                additional_files = sample(files, max_instances - len(files))
                self.file_list.extend(additional_files)


    def __len__(self):
        return int(np.ceil(self.n_files / self.batch_size))


    def on_epoch_end(self):
        self._create_file_list()
        self.shuffled = sample(self.file_list, self.n_files)


    def __getitem__(self, idx):
        batch = self.shuffled[idx * self.batch_size:(idx + 1)*self.batch_size]
        data_tiles = [self._from_pickle(x) for x in batch]
        processed = self._make_weights_labels_and_features(data_tiles, self.classes_to_augment)
        batch_x = processed[0]
        batch_y = processed[1]
        return batch_x, batch_y
    

    def _from_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data


    def _make_weights_labels_and_features(self, data_tiles, classes_to_augment):
        return _preprocess_input_data(data_tiles, self.class_weights, 
                border_width=self.border_width, classes_to_augment=classes_to_augment)


def _preprocess_input_data(data_tiles, class_weights, classes_to_augment=None, border_width=1):
    features = []
    one_hots = []
    weightings = []
    border_class = len(class_weights) - 1
    n_classes = len(class_weights)
    for tile in data_tiles:
        tile_shape = tile['data'].shape
        one_hot = np.zeros((tile_shape[1], tile_shape[2], n_classes))
        weights = np.zeros((tile_shape[1], tile_shape[2], n_classes))
        labels = tile['class_mask'][0]
        one_hot[:, :, tile['class_code']] = labels
        weights[:][labels == 1] = class_weights[tile['class_code']]
        if tile['class_code'] == 0:
            border_labels = make_border_labels(tile['class_mask'],
                    border_width=border_width)
            one_hot[:, :, border_class] = border_labels
            weights[:][border_labels[0] == 1] = class_weights[border_class]
        feature_tile = np.squeeze(tile['data'])
        feature_tile = np.swapaxes(feature_tile, 0, 2) # This is necessary b/c tf expected columns_last (GeoTiffs are columns first).
        feature_tile = np.swapaxes(feature_tile, 0, 1)
        if classes_to_augment is not None:
            if classes_to_augment[tile['class_code']]:
                # fig, ax = plt.subplots(ncols=2, nrows=2)
                # ax[0, 0].imshow(feature_tile[:, :, 18])
                # ax[0, 1].imshow(weights[:, :, tile['class_code']])
                feature_tile, one_hot, weights = _augment_data(feature_tile, one_hot, weights)
                # ax[1, 0].imshow(feature_tile[:, :, 18])
                # ax[1, 1].imshow(weights[:, :, tile['class_code']])
                # plt.show()

        features.append(feature_tile)
        one_hots.append(one_hot)
        weightings.append(weights)
    return [np.asarray(features), np.asarray(weightings)], [np.asarray(one_hots)]


def _yes_or_no():
    return choice([True, False])


def _augment_data(feature_tile, one_hot, weights):
    ''' Applies lr and ud flipping, or doesn't. '''
    if _yes_or_no():
        # Rotate the data.
        rot = np.random.randint(-25, 25)
        for i in range(feature_tile.shape[2]):
            feature_tile[:, :, i] = transform.rotate(feature_tile[:, :, i], rot, cval=0)
        for i in range(one_hot.shape[2]):
            one_hot[:, :, i] = transform.rotate(one_hot[:, :, i], rot, cval=0)
            weights[:, :, i] = transform.rotate(weights[:, :, i], rot, cval=0)
        return feature_tile, one_hot, weights
    if _yes_or_no():
        # Flip the data l-r.
        for i in range(feature_tile.shape[2]):
            feature_tile[:, :, i] = np.fliplr(feature_tile[:, :, i])
        for i in range(one_hot.shape[2]):
            one_hot[:, :, i] = np.fliplr(one_hot[:, :, i])
            weights[:, :, i] = np.fliplr(weights[:, :, i])
        return feature_tile, one_hot, weights
    if _yes_or_no():
        # Flip the data u-d.
        for i in range(feature_tile.shape[2]):
            feature_tile[:, :, i] = np.flipud(feature_tile[:, :, i])
        for i in range(one_hot.shape[2]):
            one_hot[:, :, i] = np.flipud(one_hot[:, :, i])
            weights[:, :, i] = np.flipud(weights[:, :, i])
        return feature_tile, one_hot, weights
    return feature_tile, one_hot, weights
    


def generate_unbalanced_data(training_directory='training_data/train/', border_width=2,
        batch_size=2, class_weights={}, 
        channels='all', nodata=0, n_classes=5):
    ''' Assumes data is stored in training_directory '''
    border_class = len(class_weights.keys()) - 1
    gen = DataGen(training_directory)
    while True:
        masters = []
        one_hots = []
        weightings = []
        tile_shape = None
        for _ in range(batch_size):
            tile = gen.next().copy()
            if tile_shape is None:
                tile_shape = tile['class_mask'].shape
            one_hot = np.zeros((tile_shape[1], tile_shape[2], n_classes))
            weights = np.zeros((tile_shape[1], tile_shape[2]))
            labels = tile['class_mask'][0]
            one_hot[:, :, tile['class_code']] = labels
            weights[labels == 1] = class_weights[tile['class_code']]
            if tile['class_code'] == 0:
                border_labels = make_border_labels(tile['class_mask'],
                        border_width=border_width)
                one_hot[:, :, border_class] = border_labels
                weights[border_labels[0] == 1] = class_weights[border_class]
            m = np.squeeze(tile['data'])
            m = np.swapaxes(m, 0, 2)
            m = np.swapaxes(m, 0, 1)
            masters.append(m)
            one_hots.append(one_hot)
            weightings.append(weights)

        yield np.asarray(masters), np.asarray(one_hots), np.asarray(weightings)


def generate_training_data(training_directory, threshold=None, sigma=None,
        w0=None, class_weights={}, channels='all', nodata=0, n_classes=5):
    ''' Assumes data is stored in training_directory
    in subdirectories labeled class_n_train with n the class code '''
    class_dirs = [os.path.join(training_directory, x) for x in os.listdir(training_directory)]
    if not len(class_dirs):
        class_dirs = [training_directory]
    generators = []
    border_class = len(class_weights.keys())
    for d in class_dirs:
        generators.append(DataGen(d))
    while True:
        masters = []
        one_hots = []
        weightings = []
        tile_shape = None
        for _ in range(2):
            data_tiles = []
            weighting_dict = {}
            count_dict = {}
            for gen in generators:
                out = gen.next().copy()
                if tile_shape is None:
                    tile_shape = out['class_mask'].shape
                data_tiles.append(out)
                n_samples = len(np.where(out['class_mask'] != nodata)[0])
                weighting_dict[out['class_code']] = n_samples
                count_dict[out['class_code']] = n_samples

            maxx = max(weighting_dict.values())
            for key in weighting_dict:
                weighting_dict[key] = maxx / weighting_dict[key]
            
            for tile in data_tiles:
                one_hot = np.zeros((tile_shape[1], tile_shape[2], n_classes))
                weights = np.zeros((tile_shape[1], tile_shape[2]))
                labels = tile['class_mask']
                one_hot[:, :, tile['class_code']] = labels
                weights[labels == 1] = class_weights[tile['class_code']]
                if tile['class_code'] == 0:
                    border_labels = make_border_labels(tile['class_mask'], border_width=2)
                    one_hot[:, :, border_class] = border_labels
                    weights[border_labels == 1] = class_weights[border_class]

                m = np.squeeze(tile['data'])
                m = np.swapaxes(m, 0, 2)
                masters.append(m)
                one_hots.append(one_hot)
                weightings.append(weights)

        yield np.asarray(masters), np.asarray(masks), np.asarray(weightings)


if __name__ == '__main__':
    pass
