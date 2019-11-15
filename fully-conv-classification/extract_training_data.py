import numpy as np
import numpy.ma as ma
import os
import time
import pickle
import warnings
import argparse
import pdb
import matplotlib.pyplot as plt

from glob import glob
from random import sample, shuffle, choice
from scipy.ndimage.morphology import distance_transform_edt
from rasterio import open as rasopen, band
from rasterio.errors import RasterioIOError
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage import transform
from sat_image.warped_vrt import warp_single_image
from multiprocessing import Pool 
from collections import defaultdict

from runspec import (landsat_rasters, climate_rasters, mask_rasters, assign_shapefile_class_code,
        assign_shapefile_year, cdl_crop_values, cdl_non_crop_values)
from data_utils import load_raster, paths_map_multiple_scenes, stack_rasters, stack_rasters_multiprocess, download_from_pr, paths_mapping_single_scene, mean_of_three, median_of_three
from shapefile_utils import get_shapefile_path_row, mask_raster_to_shapefile, filter_shapefile_overlapping, mask_raster_to_features


def distance_map(mask):
    mask = mask.copy().astype(bool)
    mask = ~mask # make the non-masked areas masked
    distances = distance_transform_edt(mask) # ask where the closest masked pixel is
    return distances


class DataTile(object):

    def __init__(self, data, one_hot, class_code, cdl_mask):
        self.dict = {}
        self.dict['data'] = data
        self.dict['one_hot'] = one_hot
        self.dict['class_code'] = class_code
        self.dict['cdl'] = cdl_mask

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
            raise ValueError()


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


def concatenate_fmasks(image_directory, class_mask, class_mask_geo, nodata=0, target_directory=None):
    ''' 
    ``Fmasks'' are masks of clouds and water. We don't want clouds/water in
    the training set, so this function gets all the fmasks for a landsat
    scene (contained in image_directory), and merges them into one raster. 
    They may not be the same size, so warp_vrt is used to make them align. 
    '''
    class_mask = class_mask.copy()
    paths = []
    for dirpath, dirnames, filenames in os.walk(image_directory):
        for f in filenames:
            for suffix in mask_rasters():
                if f.endswith(suffix):
                    pth = os.path.join(dirpath, f)
                    paths.append(pth)
    for fmask_file in paths:
        fmask, _ = load_raster(fmask_file)
        # clouds, water present where fmask == 1.
        try:
            class_mask = ma.masked_where(fmask == 1, class_mask)
        except (ValueError, IndexError) as e:
            fmask = warp_single_image(fmask_file, class_mask_geo)
            class_mask = ma.masked_where(fmask == 1, class_mask)

    return class_mask


def reproject_if_needed(source, target):
    with rasopen(target, 'r') as dst:
        dst_crs = dst.meta['crs']

    with rasopen(source) as src:
        if src.meta['crs'] == dst_crs:
            return source
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasopen(source, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=band(src, i),
                    destination=band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return source


def extract_training_data_over_path_row(test_train_shapefiles, path, row, year, image_directory,
        training_data_root_directory, n_classes, assign_shapefile_class_code, path_map_func=None,
        preprocessing_func=None, tile_size=608):

    if path_map_func is None:
        path_map_func = paths_map_multiple_scenes

    if not isinstance(test_train_shapefiles, dict):
        raise ValueError("expected dict, got {}".format(type(test_train_shapefiles)))
    
    path_row_year = str(path) + '_' + str(row) +  '_' + str(year)
    image_path = os.path.join(image_directory, path_row_year)
    if not os.path.isdir(image_path):
        download_from_pr(path, row, year, image_directory)
    image_path_maps = path_map_func(image_path)
    mask_file = _random_tif_from_directory(image_path)
    mask, mask_meta = load_raster(mask_file)
    mask = np.zeros_like(mask).astype(np.int)
    cdl_path = os.path.join(image_path, 'cdl_mask.tif')
    cdl_raster, cdl_meta = load_raster(cdl_path)
    if mask.shape != cdl_raster.shape:
        cdl_raster = warp_single_image(cdl_path, mask_meta)
    cdl_raster = np.swapaxes(cdl_raster, 0, 2)
    try:
        image_stack = stack_rasters_multiprocess(image_path_maps, target_geo=mask_meta, target_shape=mask.shape)
        image_stack = np.swapaxes(image_stack, 0, 2)
    except RasterioIOError as e:
        print("Redownload images for", path_row_year)
        print(e)
        return
    for key, shapefiles in test_train_shapefiles.items():
        if key.lower() not in ('test', 'train'):
            raise ValueError("expected key to be one of case-insenstive {test, train},\
            got {}".format(key))

        training_data_directory = os.path.join(training_data_root_directory, key)
        first = True
        class_labels = None
        for f in shapefiles:
            class_code = assign_shapefile_class_code(f)
            print(f, class_code)
            out, _ = mask_raster_to_shapefile(f, mask_file, return_binary=False)
            if first:
                class_labels = out
                class_labels[~class_labels.mask] = class_code
                first = False
            else:
                class_labels[~out.mask] = class_code
        class_labels = concatenate_fmasks(image_path, class_labels, mask_meta) 
        class_labels = np.swapaxes(class_labels, 0, 2)
        class_labels = np.squeeze(class_labels)
        tiles_y, tiles_x = _target_indices_from_class_labels(class_labels, tile_size)
        _save_training_data_from_indices(image_stack, class_labels, cdl_raster,
                training_data_directory, n_classes, tiles_x, tiles_y, tile_size)


def _target_indices_from_class_labels(class_labels, tile_size):
    where = np.nonzero(~class_labels.mask)
    max_y = np.max(where[0])
    min_y = np.min(where[0])
    max_x = np.max(where[1])
    min_x = np.min(where[1])
    max_y += (tile_size - ((max_y - min_y) % tile_size))
    max_x += (tile_size - ((max_x - min_x) % tile_size))
    tiles_y = range(min_y, max_y, tile_size)
    tiles_x = range(min_x, max_x, tile_size)
    return tiles_y, tiles_x


def _assign_class_code_to_tile(class_label_tile):
    if np.any(class_label_tile == 3):
        return 3
    if np.all(class_label_tile != 0):
        unique, unique_count = np.unique(class_label_tile, return_counts=True)
        unique = unique[:-1] # assume np.ma.masked is last.
        unique_count = unique_count[:-1]
        return unique[np.argmax(unique_count)]
    # if a tile has any irrigated pixels, return 0.
    return 0


def _save_training_data_from_indices(image_stack, class_labels, cdl_raster,
        training_data_directory, n_classes, indices_y, indices_x, tile_size):
    out = []
    for i in indices_x:
        for j in indices_y:
            class_label_tile = class_labels[i:i+tile_size, j:j+tile_size]
            shape = class_label_tile.shape
            if np.all(class_label_tile.mask):
                continue
            if (shape[0], shape[1]) != (tile_size, tile_size):
                continue
            class_code = _assign_class_code_to_tile(class_label_tile)
            sub_one_hot = _one_hot_from_labels(class_label_tile, n_classes)
            sub_cdl = cdl_raster[i:i+tile_size, j:j+tile_size, :]
            sub_image_stack = image_stack[i:i+tile_size, j:j+tile_size, :]
            sub_image_stack = image_stack[i:i+tile_size, j:j+tile_size, :]
            dt = DataTile(sub_image_stack, sub_one_hot, class_code, sub_cdl)
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


def _random_tif_from_directory(image_directory):

    bleh = os.listdir(image_directory)
    for d in bleh:
        if os.path.isdir(os.path.join(image_directory, d)):
            tiffs = glob(os.path.join(os.path.join(image_directory, d), "*.TIF"))
            tiffs = [tif for tif in tiffs if 'BQA' not in tif]
            break
    shuffle(tiffs)
    return tiffs[0]


def min_data_tiles_to_cover_labels(shapefiles, path, row, year, image_directory, tile_size=608):
    path_row_year = "_".join([str(path), str(row), str(year)])
    image_directory = os.path.join(image_directory, path_row_year)
    mask_file = _random_tif_from_directory(image_directory)
    mask, mask_meta = load_raster(mask_file)
    mask = np.zeros_like(mask).astype(np.int)
    first = True
    class_labels = None
    if not isinstance(shapefiles, list):
        shapefiles = [shapefiles]
    for f in shapefiles:
        class_code = assign_shapefile_class_code(f)
        out, _ = mask_raster_to_shapefile(f, mask_file, return_binary=False)
        if first:
            class_labels = out
            class_labels[~class_labels.mask] = class_code
            first = False
        else:
            class_labels[~out.mask] = class_code
    class_labels = concatenate_fmasks(image_directory, class_labels, mask_meta) 
    where = np.nonzero(~class_labels.mask[0])
    max_y = np.max(where[0])
    min_y = np.min(where[0])
    max_x = np.max(where[1])
    min_x = np.min(where[1])
    frac = np.count_nonzero(~class_labels.mask)/(class_labels.shape[1]*class_labels.shape[2])

    max_y += (tile_size - ((max_y - min_y) % tile_size))
    max_x += (tile_size - ((max_x - min_x) % tile_size))

    tiles_y = range(min_y, max_y, tile_size)
    tiles_x = range(min_x, max_x, tile_size)

    plt.plot([max_x, max_x], [max_y, min_y], 'b', linewidth=2)
    plt.plot([min_x, min_x], [max_y, min_y], 'b', linewidth=2)
    plt.plot([min_x, max_x], [max_y, max_y], 'b', linewidth=2)
    plt.plot([min_x, max_x], [min_y, min_y], 'b', linewidth=2)

    y_min = [min_x] * len(tiles_y)
    y_max = [max_x] * len(tiles_y)
    for t, mn, mx in zip(tiles_y, y_min, y_max):
        plt.plot([mn, mx], [t, t], 'r')

    x_min = [min_y] * len(tiles_x)
    x_max = [max_y] * len(tiles_x)
    for t, mn, mx in zip(tiles_x, x_min, x_max):
        plt.plot([t, t], [mn, mx], 'r')

    plt.imshow(class_labels[0])
    plt.title('path/row: {} {} percent data pixels: {:.3f}'.format(path, row, frac))
    plt.colorbar()
    plt.show()


def _one_hot_from_labels(labels, n_classes):
    one_hot = np.zeros((labels.shape[0], labels.shape[1], n_classes))
    for class_code in range(n_classes):
        one_hot[:, :, class_code][labels == class_code] = 1
        # if class_code == 1: # apply border class to only irrigated pixels
        #     border_labels = make_border_labels(one_hot[:, :, 1], border_width=1)
        #     border_labels = border_labels.astype(np.uint8)
        #     one_hot[:, :, n_classes-1][border_labels == 1] = 1
    return one_hot.astype(np.int)


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


if __name__ == '__main__':
    
    image_directory = '/home/thomas/share/image_data/'
    shapefiles = glob('shapefile_data/test/*.shp') + glob('shapefile_data/train/*.shp')
    training_root_directory = '/home/thomas/share/multiclass_with_separate_fallow_directory_and_cdl/'
    if not os.path.isdir(training_root_directory):
        os.makedirs(training_root_directory)

    n_classes = 4
    done = set()

    for f in shapefiles:
        if f in done:
            continue
        test_shapefiles = all_matching_shapefiles(f, 'shapefile_data/test/', assign_shapefile_year)
        train_shapefiles = all_matching_shapefiles(f, 'shapefile_data/train/', assign_shapefile_year)
        for e in test_shapefiles + train_shapefiles:
            done.add(e)
        bs = os.path.splitext(os.path.basename(f))[0]
        _, path, row = bs[-7:].split("_")
        year = assign_shapefile_year(f)
        print("extracting data for", path, row, year)
        paths_map_func = paths_map_multiple_scenes
        test_train_shapefiles = {'test':test_shapefiles, 'train':train_shapefiles}
        extract_training_data_over_path_row(test_train_shapefiles, path, row, year, image_directory,
                training_root_directory, n_classes, assign_shapefile_class_code, path_map_func=paths_map_func)

