import numpy as np
import numpy.ma as ma
import os
import time
import pickle
from matplotlib.pyplot import imshow, show, subplots, colorbar
import warnings
import pdb

from glob import glob
from random import sample, shuffle, choice
from scipy.ndimage.morphology import distance_transform_edt
from rasterio import open as rasopen
from rasterio.errors import RasterioIOError
from skimage import transform
from sat_image.warped_vrt import warp_single_image
from tensorflow.keras.utils import Sequence
from multiprocessing import Pool
from collections import defaultdict

from runspec import mask_rasters
from data_utils import load_raster, paths_map_multiple_scenes, stack_rasters, stack_rasters_multiprocess
from shapefile_utils import get_shapefile_path_row, mask_raster_to_shapefile, filter_shapefile_overlapping, mask_raster_to_features


def distance_map(mask):
    mask = mask.copy().astype(bool)
    mask = ~mask # make the non-masked areas masked
    distances = distance_transform_edt(mask) # ask where the closest masked pixel is
    return distances


class DataTile(object):

    def __init__(self, data, one_hot, class_code):
        self.dict = {}
        self.dict['data'] = data.astype(np.float32)
        self.dict['one_hot'] = one_hot
        self.dict['class_code'] = class_code
        # Need to split the data into separate classes to play with class balance.

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


def _pickle_datatile(datatile, training_directory):
        template = os.path.join(training_directory,
                'class_{}_data/'.format(datatile.dict['class_code']))
        if not os.path.isdir(template):
            os.mkdir(template)
        outfile = os.path.join(template, str(time.time()) + ".pkl")
        if not os.path.isfile(outfile):
            with open(outfile, 'wb') as f:
                pickle.dump(datatile.dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pass


def concatenate_fmasks(image_directory, class_mask, class_mask_geo, nodata=0):
    ''' 
    ``Fmasks'' are masks of clouds and water. We don't want clouds/water in
    the training set, so this function gets all the fmasks for a landsat
    scene (contained in image_directory), and merges them into one raster. 
    They may not be the same size, so warp_vrt is used to make them align. 
    '''
    paths = []
    for dirpath, dirnames, filenames in os.walk(image_directory):
        for f in filenames:
            for suffix in mask_rasters():
                if f.endswith(suffix):
                    paths.append(os.path.join(dirpath, f))
    paths = [p for p in paths if 'water' not in p]
    for fmask_file in paths:
        fmask, _ = load_raster(fmask_file)
        # why is water being misregistered?
        # clouds, water present where fmask == 1.
        try:
            class_mask = ma.masked_where(fmask == 1, class_mask)
        except (ValueError, IndexError) as e:
            fmask = warp_single_image(fmask_file, class_mask_geo)
            class_mask = ma.masked_where(fmask == 1, class_mask)

    return class_mask


def extract_training_data_v2(split_shapefile_directory, image_directory,
        training_data_directory, assign_shapefile_year, assign_shapefile_class_code, n_classes=5):

    split_shapefiles = [f for f in glob(os.path.join(split_shapefile_directory, "*.shp"))]

    done = set()

    total_time = 0

    for counter, shapefile in enumerate(split_shapefiles):
        begin_time = time.time()
        if shapefile in done:
            continue
        _, path, row = os.path.splitext(shapefile)[0][-7:].split('_')
        year = assign_shapefile_year(shapefile)
        path_row_year = path + '_' + row +  '_' + str(year)
        print("Extracting data for", path_row_year)
        shapefiles_over_same_path_row = all_matching_shapefiles(shapefile,
                split_shapefile_directory, assign_shapefile_year)
        done.update(shapefiles_over_same_path_row)
        image_path = os.path.join(image_directory, path_row_year)
        if not os.path.isdir(image_path):
            print('hooby dooby')
            continue
        image_path_map = paths_map_multiple_scenes(os.path.join(image_directory, path_row_year))
        mask_file = image_path_map['B1.TIF'][0]
        mask, mask_meta = load_raster(mask_file)
        try:
            image_stack = stack_rasters_multiprocess(image_path_map, target_geo=mask_meta, target_shape=mask.shape)
        except RasterioIOError as e:
            print("Redownload images for", path_row_year)
            # TODO: remove corrupted file and redownload images.
            continue
        mask = np.zeros_like(mask).astype(np.int)
        fmask = concatenate_fmasks(os.path.join(image_directory, path_row_year), mask,
                mask_meta) 
        if fmask.mask.all():
            print("All pixels covered by cloud for {}".format(path_row_year))
            continue

        one_hots = []
        class_codes = []
                
        for f in shapefiles_over_same_path_row:
            class_code = assign_shapefile_class_code(f)
            out = _one_hot_from_shapefile(f, mask_file, class_code, n_classes)
            if out is not None:
                one_hots.append(out)
                class_codes.append(class_code)
        # b/c tf expects columns first, we swapaxes here.
        one_hot_copy = []
        for one_hot, class_code in zip(one_hots, class_codes):
            for i in range(n_classes):
                one_hot[i, :, :][fmask.mask[0]] = ma.masked # why can't i vectorize this
            if class_code == 0: # apply border class to only irrigated pixels
                border_labels = make_border_labels(one_hot[i, :, :], border_width=1)
                border_labels.astype(bool)
                one_hot[n_classes-1, :, :] = border_labels
            one_hot = np.swapaxes(one_hot, 0, 2)  
            one_hot_copy.append(one_hot) 
        image_stack = np.swapaxes(image_stack, 0, 2)
        _iterate_over_raster_v2(image_stack, one_hot_copy, class_codes, training_data_directory)
        end_time = time.time()
        diff = end_time - begin_time
        total_time += diff
        print('single iteration time:', diff, 'avg.', total_time / (counter + 1))


def _iterate_over_raster_v2(raster, one_hots, class_codes, training_directory, tile_size=608):
    out = []
    # ... could rewrite it in cython.
    for i in range(0, raster.shape[0]-tile_size, tile_size):
        for j in range(0, raster.shape[1]-tile_size, tile_size):
            for one_hot, class_code in zip(one_hots, class_codes):
                sub_one_hot = one_hot[i:i+tile_size, j:j+tile_size, :]
                if not _check_dimensions_and_min_pixels(sub_one_hot, tile_size):
                    continue
                sub_raster = raster[i:i+tile_size, j:j+tile_size, :]
                dt = DataTile(sub_raster, sub_one_hot, class_code)
                out.append(dt)
    if len(out):
        with Pool() as pool:
            td = [training_directory]*len(out)
            pool.starmap(_pickle_datatile, zip(out, td))


def _one_hot_from_shapefile(shapefile, mask_file, shapefile_class_code, n_classes):
    class_labels, _ = mask_raster_to_shapefile(shapefile, mask_file, return_binary=False)
    if class_labels.mask.all():
        return None
    one_hot = _one_hot_from_labels(class_labels, shapefile_class_code, n_classes)
    return one_hot


def _one_hot_from_labels(labels, class_code, n_classes):
    out = np.zeros((n_classes, labels.shape[1], labels.shape[2]))
    out[class_code, :, :][~labels.mask[0]] = 1
    return out.astype(np.int)


def _check_dimensions_and_min_pixels(sub_one_hot, tile_size):
    # 200 is the minimum amount of pixels required to save the data.
    if sub_one_hot.shape[0] != tile_size or sub_one_hot.shape[1] != tile_size:
        return False
    if len(np.nonzero(sub_one_hot)[0]) < 200:
        return False
    return True


def extract_training_data(shapefile_directory, image_directory,
        training_directory, save=True, tile_size=608,
        assign_shapefile_year=None, assign_shapefile_class_code=None,
        min_pixels=500, fmask=True, n_classes=4, nodata=0, augment_dict={}):

    if isinstance(assign_shapefile_year, type(None)):
        raise ValueError("Please provide a function to assign shapefile year.")
    if isinstance(assign_shapefile_class_code, type(None)):
        raise ValueError("Please provide a function to assign shapefile class code.")

    pixel_dict = {} # pixel dict counts number of pixels present in each class.
    for class_code in range(n_classes):
        pixel_dict[class_code] = 0 
    done = set()
    all_shapefiles = [f for f in glob(os.path.join(shapefile_directory, "*.shp"))]
    for f in all_shapefiles:
        if f not in done:
            done.add(f)
            all_matches = all_matching_shapefiles(f, shapefile_directory, assign_shapefile_year) # get all shapefiles
            # in the same path / row / year
            for match in all_matches:
                done.add(match)
            p, r = get_shapefile_path_row(f) #TODO: error checking on this function.
            year = assign_shapefile_year(f)
            suffix = '{}_{}_{}'.format(p, r, year) 
            if not os.path.isdir(os.path.join(image_directory, suffix)):
                # TODO: Figure out why the warning isn't working.
                print("Images for {} not in given image directory ({}). Skipping extraction of data for following shapefiles: {}".format(suffix, image_directory, [os.path.basename(x) for x in all_matches]))
                continue
            paths_mapping = paths_map(os.path.join(image_directory, suffix)) 
            try:
                master = stack_rasters(paths_mapping, p, r, year) 
                #    paths_mapping
            except Exception as e:
                print(e)
                print("Bad image data in", suffix)
                continue
            mask_file = paths_mapping['B1.TIF'][0] #TODO: this shouldn't be hardcoded.
            masks = []
            # TODO: Only warp fmasks/load them into memory once. 
            for match in all_matches:
                cc = assign_shapefile_class_code(match)
                if cc is None:
                    raise ValueError("Shapefile {} not provided with a class code.".format(os.path.basename(match)))
                msk, mask_meta = mask_raster_to_shapefile(match, mask_file, return_binary=True)
                if fmask:
                    msk = concatenate_fmasks(os.path.join(image_directory, suffix), msk,
                            mask_meta, nodata=nodata) 
                dm = DataMask(msk, cc) # a binary mask that has a class_code attributed to it.
                masks.append(dm)
                print("Extracting data for {}. CC: {}. Year: {}".format(os.path.basename(match), cc,
                    year))

            pixel_dict = _iterate_over_raster(master, masks, pixel_dict, 
                    tile_size=tile_size, save=save, min_pixels=min_pixels,
                    training_directory=training_directory)
            print("{} of {} shapefiles done. ".format(len(done), len(all_shapefiles)))

    return pixel_dict


def _iterate_over_raster(raster, datamasks, pixel_dict, tile_size=608, 
        save=True, training_directory=None, min_pixels=None):
    step = tile_size 
    for i in range(0, raster.shape[1]-tile_size, step):
        for j in range(0, raster.shape[2]-tile_size, step):
            sub_raster = raster[:, i:i+tile_size, j:j+tile_size]
            for datamask in datamasks:
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


def all_matching_shapefiles(to_match, shapefile_directory, assign_shapefile_year):
    out = []
    pr = get_shapefile_path_row(to_match)
    year = assign_shapefile_year(to_match)
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if get_shapefile_path_row(f) == pr and assign_shapefile_year(f) == year:
                out.append(f)
    return out


def make_border_labels(mask, border_width):
    ''' Border width: Pixel width. '''
    dm = distance_map(mask)
    dm[dm > border_width] = 0
    return dm


class SatDataSequence(Sequence):

    def __init__(self, data_directory, batch_size, class_weights={},
            border_width=1, n_classes=5, classes_to_augment=None):
        self.data_directory = data_directory
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.batch_size = batch_size
        self._no_augment = classes_to_augment is None
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
                    shuffle(files)
                additional_files = sample(files, max_instances - len(files))
                self.file_list.extend(additional_files)


    def __len__(self):
        return int(np.ceil(self.n_files / self.batch_size))


    def on_epoch_end(self):
        self._create_file_list()
        shuffle(self.file_list)
        self.shuffled = self.file_list


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
        return self._preprocess_input_data(data_tiles, self.class_weights, 
                classes_to_augment=classes_to_augment)


    def _preprocess_input_data(self, data_tiles, class_weights, classes_to_augment=None):
        features = []
        one_hots = []
        weight_list = []
        for tile in data_tiles:
            data = tile['data']
            one_hot = tile['one_hot'].astype(np.int)
            weights = np.zeros_like(one_hot)
            class_code = tile['class_code']
            weights[:][one_hot[:, :, class_code]] = class_weights[class_code]
            if class_code == 0:
                weights[:][one_hot[:, :, self.n_classes-1]] = class_weights[self.n_classes-1]
            if not self._no_augment:
                if classes_to_augment[tile['class_code']]:
                    data, one_hot, weights = _augment_data(data, one_hot, weights)
            features.append(data)
            one_hots.append(one_hot)
            weight_list.append(weights)
        return [np.asarray(features), np.asarray(weight_list)], [np.asarray(one_hots)]


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
        feature_tile = np.swapaxes(feature_tile, 0, 2) # This is necessary b/c tf expects columns_last (GeoTiffs are columns first).
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


def _flip_lr(feature_tile, one_hot, weights):
    for i in range(feature_tile.shape[2]):
        feature_tile[:, :, i] = np.fliplr(feature_tile[:, :, i])
    for i in range(one_hot.shape[2]):
        one_hot[:, :, i] = np.fliplr(one_hot[:, :, i])
        weights[:, :, i] = np.fliplr(weights[:, :, i])
    return feature_tile, one_hot, weights


def _flip_ud(feature_tile, one_hot, weights):
    for i in range(feature_tile.shape[2]):
        feature_tile[:, :, i] = np.flipud(feature_tile[:, :, i])
    for i in range(one_hot.shape[2]):
        one_hot[:, :, i] = np.flipud(one_hot[:, :, i])
        weights[:, :, i] = np.flipud(weights[:, :, i])
    return feature_tile, one_hot, weights


def _rotate(feature_tile, one_hot, weights):
    # Rotate data.
    rot = np.random.randint(-25, 25)
    for i in range(feature_tile.shape[2]):
        feature_tile[:, :, i] = transform.rotate(feature_tile[:, :, i], rot, cval=0)
    for i in range(one_hot.shape[2]):
        one_hot[:, :, i] = transform.rotate(one_hot[:, :, i], rot, cval=0)
        weights[:, :, i] = transform.rotate(weights[:, :, i], rot, cval=0)
    return feature_tile, one_hot, weights


def _flip_lr_ud(feature_tile, one_hot, weights):
    feature_tile, one_hot, weights = _flip_lr(feature_tile, one_hot, weights)
    feature_tile, one_hot, weights = _flip_ud(feature_tile, one_hot, weights)
    return feature_tile, one_hot, weights


def _do_nothing(feature_tile, one_hot, weights):
    return feature_tile, one_hot, weights


def _augment_data(feature_tile, one_hot, weights):
    ''' Applies rotation | lr | ud | lr_ud | flipping, or doesn't. '''
    possible_augments = [_flip_ud, _flip_lr, _flip_lr_ud, _do_nothing]
    return choice(possible_augments)(feature_tile, one_hot, weights)


if __name__ == '__main__':
    pass
