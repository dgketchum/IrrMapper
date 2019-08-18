import numpy as np
import numpy.ma as ma
import os
import time
import pickle
import warnings
import pdb
import matplotlib.pyplot as plt

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
from data_utils import load_raster, paths_map_multiple_scenes, stack_rasters, stack_rasters_multiprocess, download_from_pr
from shapefile_utils import get_shapefile_path_row, mask_raster_to_shapefile, filter_shapefile_overlapping, mask_raster_to_features


def distance_map(mask):
    mask = mask.copy().astype(bool)
    mask = ~mask # make the non-masked areas masked
    distances = distance_transform_edt(mask) # ask where the closest masked pixel is
    return distances


class DataTile(object):

    def __init__(self, data, one_hot, weights, class_code):
        self.dict = {}
        self.dict['data'] = data
        self.dict['one_hot'] = one_hot
        self.dict['class_code'] = class_code
        self.dict['weights'] = weights
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
    for fmask_file in paths:
        fmask, _ = load_raster(fmask_file)
        # clouds, water present where fmask == 1.
        try:
            class_mask = ma.masked_where(fmask == 1, class_mask)
        except (ValueError, IndexError) as e:
            fmask = warp_single_image(fmask_file, class_mask_geo)
            class_mask = ma.masked_where(fmask == 1, class_mask)

    return class_mask


def extract_training_data_multiple_classes_per_instance(split_shapefile_directory, image_directory,
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
            download_from_pr(path, row, year, image_directory)
            continue
        image_path_map = paths_map_multiple_scenes(os.path.join(image_directory, path_row_year))
        try:
            mask_file = image_path_map['B1.TIF'][0]
        except IndexError:
            os.rmdir(os.path.join(image_directory, path_row_year))
            download_from_pr(path, row, year, image_directory)
            image_path_map = paths_map_multiple_scenes(os.path.join(image_directory, path_row_year))
            mask_file = image_path_map['B1.TIF'][0]

        mask, mask_meta = load_raster(mask_file)
        mask = np.zeros_like(mask).astype(np.int)
        fmask = concatenate_fmasks(os.path.join(image_directory, path_row_year), mask,
                mask_meta) 
        if fmask.mask.all():
            print("All pixels covered by cloud for {}".format(path_row_year))
            continue

        first = True
        class_labels = None
        for f in shapefiles_over_same_path_row:
            class_code = assign_shapefile_class_code(f)
            print(f, class_code)
            out, _ = mask_raster_to_shapefile(f, mask_file, return_binary=False)
            if first:
                class_labels = out
                class_labels[~class_labels.mask] = class_code
                first = False
            else:
                class_labels[~out.mask] = class_code
        try:
            image_stack = stack_rasters_multiprocess(image_path_map, target_geo=mask_meta, target_shape=mask.shape)
        except RasterioIOError as e:
            print("Redownload images for", path_row_year)
            # TODO: remove corrupted file and redownload images.
            continue
        class_labels[fmask.mask] = ma.masked # well, I didn't fmask the data.
        image_stack = np.swapaxes(image_stack, 0, 2)
        class_labels = np.swapaxes(class_labels, 0, 2)
        class_labels = np.squeeze(class_labels)
        _save_training_data_multiple_classes(image_stack, class_labels,
                training_data_directory, n_classes)
        end_time = time.time()
        diff = end_time - begin_time
        total_time += diff
        print('single iteration time:', diff, 'avg.', total_time / (counter + 1))


def _save_training_data_multiple_classes(image_stack, class_labels, training_data_directory, n_classes):
    tile_size = 608
    out = []
    class_code = 7 # dummy...
    for i in range(0, image_stack.shape[0]-tile_size, tile_size):
        for j in range(0, image_stack.shape[1]-tile_size, tile_size):
            class_label_tile = class_labels[i:i+tile_size, j:j+tile_size]
            if np.all(class_label_tile.mask == True):
                continue
            sub_one_hot = _one_hot_from_labels_mc(class_label_tile, n_classes)
            weights = _weights_from_one_hot(sub_one_hot, n_classes)
            sub_image_stack = image_stack[i:i+tile_size, j:j+tile_size, :]
            dt = DataTile(sub_image_stack, sub_one_hot, weights, class_code)
            out.append(dt)
            if len(out) > 50:
                with Pool() as pool:
                    td = [training_data_directory]*len(out)
                    pool.starmap(_pickle_datatile, zip(out, td))
                out = []
    if len(out):
        with Pool() as pool:
            td = [training_data_directory]*len(out)
            pool.starmap(_pickle_datatile, zip(out, td))
            out = []


def _one_hot_from_labels_mc(labels, n_classes):
    one_hot = np.zeros((labels.shape[0], labels.shape[1], n_classes))
    for class_code in range(n_classes):
        one_hot[:, :, class_code][labels == class_code] = 1
        if class_code == 0: # apply border class to only irrigated pixels
            border_labels = make_border_labels(one_hot[:, :, 0], border_width=1)
            border_labels.astype(np.int)
            one_hot[:, :, n_classes-1] = border_labels
    return one_hot.astype(np.int)


def extract_training_data_single_class_per_instance(split_shapefile_directory, image_directory,
        training_data_directory, assign_shapefile_year, assign_shapefile_class_code,
        offline_augmentation_dict=None, n_classes=5):

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
            download_from_pr(path, row, year, image_directory)
            continue
        image_path_map = paths_map_multiple_scenes(os.path.join(image_directory, path_row_year))
        try:
            mask_file = image_path_map['B1.TIF'][0]
        except IndexError:
            os.rmdir(os.path.join(image_directory, path_row_year))
            download_from_pr(path, row, year, image_directory)
            image_path_map = paths_map_multiple_scenes(os.path.join(image_directory, path_row_year))
            mask_file = image_path_map['B1.TIF'][0]

        mask, mask_meta = load_raster(mask_file)
        mask = np.zeros_like(mask).astype(np.int)
        fmask = concatenate_fmasks(os.path.join(image_directory, path_row_year), mask,
                mask_meta) 
        if fmask.mask.all():
            print("All pixels covered by cloud for {}".format(path_row_year))
            continue
        first = True
        class_labels = None
        for f in shapefiles_over_same_path_row:
            class_code = assign_shapefile_class_code(f)
            if offline_augmentation_dict[class_code] == 0:
                continue
            print(f, class_code)
            out, _ = mask_raster_to_shapefile(f, mask_file, return_binary=False)
            if first:
                class_labels = out
                class_labels[~class_labels.mask] = class_code
                first = False
            else:
                class_labels[~out.mask] = class_code
        if class_labels is None:
            print("no extra augmentation for", path_row_year)
            continue
        try:
            image_stack = stack_rasters_multiprocess(image_path_map, target_geo=mask_meta, target_shape=mask.shape)
        except RasterioIOError as e:
            print("Redownload images for", path_row_year)
            # TODO: remove corrupted file and redownload images.
            continue
        class_labels[fmask.mask] = ma.masked # well, I didn't fmask the data.
        image_stack = np.swapaxes(image_stack, 0, 2)
        class_labels = np.swapaxes(class_labels, 0, 2)
        class_labels = np.squeeze(class_labels)
        _save_training_data_offline_augmentation(image_stack, class_labels,
                training_data_directory, n_classes, offline_augmentation_dict)
        end_time = time.time()
        diff = end_time - begin_time
        total_time += diff
        print('single iteration time:', diff, 'avg.', total_time / (counter + 1))


def _save_training_data_offline_augmentation(image_stack, class_labels,
        training_data_directory, n_classes, offline_augmentation_dict):
    unique = np.unique(class_labels)
    unique = unique[~unique.mask]
    tile_size = 608
    for class_code in unique:
        out = []
        augmentation_step = offline_augmentation_dict[class_code]
        for i in range(0, image_stack.shape[0]-tile_size, augmentation_step):
            for j in range(0, image_stack.shape[1]-tile_size, augmentation_step):
                class_label_tile = class_labels[i:i+tile_size, j:j+tile_size]
                if not _check_dimensions_and_min_pixels(class_label_tile, class_code, tile_size):
                    continue
                sub_one_hot = _one_hot_from_labels(class_label_tile, class_code, n_classes)
                weights = _weights_from_one_hot(sub_one_hot, n_classes)
                sub_image_stack = image_stack[i:i+tile_size, j:j+tile_size, :]
                dt = DataTile(sub_image_stack, sub_one_hot, weights, class_code)
                out.append(dt)
                if len(out) > 50:
                    with Pool() as pool:
                        td = [training_data_directory]*len(out)
                        pool.starmap(_pickle_datatile, zip(out, td))
                    out = []
        if len(out):
            with Pool() as pool:
                td = [training_data_directory]*len(out)
                pool.starmap(_pickle_datatile, zip(out, td))
                out = []


def _weights_from_one_hot(one_hot, n_classes):
    weights = np.zeros_like(one_hot)
    tmp = np.sum(one_hot, 2)
    for i in range(n_classes):
        weights[:, :, i] = tmp
    return weights.astype(bool)


def _one_hot_from_shapefile(shapefile, mask_file, shapefile_class_code, n_classes):
    class_labels, _ = mask_raster_to_shapefile(shapefile, mask_file, return_binary=False)
    if class_labels.mask.all():
        return None
    one_hot = _one_hot_from_labels(class_labels, shapefile_class_code, n_classes)
    return one_hot


def _one_hot_from_labels(labels, class_code, n_classes):
    one_hot = np.zeros((labels.shape[0], labels.shape[1], n_classes))
    one_hot[:, :, class_code][labels == class_code] = 1
    if class_code == 0: # apply border class to only irrigated pixels
        border_labels = make_border_labels(one_hot[:, :, 0], border_width=1)
        border_labels.astype(np.int)
        one_hot[:, :, n_classes-1] = border_labels
    return one_hot.astype(np.int)


def _check_dimensions_and_min_pixels(sub_one_hot, class_code, tile_size):
    # 200 is the minimum amount of pixels required to save the data.
    if sub_one_hot.shape[0] != tile_size or sub_one_hot.shape[1] != tile_size:
        return False
    xx = np.where(sub_one_hot == class_code)
    if len(xx[0]) == 0:
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
