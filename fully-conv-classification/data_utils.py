import os
import geopandas as gpd
import json
from fiona import open as fopen
from glob import glob
from lxml import html
from requests import get
from copy import deepcopy
from numpy import zeros, asarray, array, reshape, nan, sqrt, std
from shapely.geometry import shape
from collections import defaultdict
from rasterio import float32, open as rasopen
from rasterio.mask import mask
from pickle import load
from prepare_images import ImageStack
from shapefile_utils import get_features
from sat_image.warped_vrt import warp_single_image
from runspec import landsat_rasters, static_rasters, climate_rasters

WRS2 = '../spatial_data/wrs2_descending_usa.shp'

def paths_map(image_directory, satellite=8):
    ''' Recursively get all rasters in image_directory
    and its subdirectories, and adds them to band_map. '''
    band_map = defaultdict(list)
    for band in landsat_rasters()[satellite]:
        band_map[band] = []
    for band in static_rasters():
        band_map[band] = []
    for band in climate_rasters():
        band_map[band] = []

    extensions = (".tif", ".TIF")
    for dirpath, dirnames, filenames in os.walk(image_directory):
        for f in filenames:
            if any(ext in f for ext in extensions):
                for band in band_map:
                    if f.endswith(band):
                        band_map[band].append(os.path.join(dirpath, f))

    for band in band_map:
        band_map[band] = sorted(band_map[band]) # ensures ordering within bands - sort by time.

    return band_map


def stack_rasters(paths_map, path, row, year):
    first = True
    stack = None
    num_rasters = 0
    for key in paths_map:
        num_rasters += len(paths_map[key])
    j = 0
    for feat in sorted(paths_map.keys()): # ensures the stack is in the same order each time.
        # Ordering within bands is assured by sorting the list that
        # each band corresponding to, as that's sorting by date.
        feature_rasters = paths_map[feat] # maps bands to their location in filesystem.
        for feature_raster in feature_rasters:
            with rasopen(feature_raster, mode='r') as src:
                arr = src.read()
                raster_geo = src.meta.copy()
            if first:
                first_geo = raster_geo.copy()
                empty = zeros((num_rasters, arr.shape[1], arr.shape[2]), float32)
                stack = empty
                stack[j, :, :] = arr
                j += 1
                first = False
            else:
                try:
                    stack[j, :, :] = arr
                    j += 1
                except ValueError: 
                    arr = warp_single_image(feature_raster, first_geo)
                    stack[j, :, :] = arr
                    j += 1
    return stack


def create_master_raster(paths_map, path, row, year, raster_directory, mean_map=None, 
        stddev_map=None):
    """ Creates a master raster with depth given by the organization of the
    paths_map. Paths map is a dictionary of lists, with keys the band names 
    (B1, B2...) and values the paths of the images in the filesystem 
    corresponding to that band. """
    fname = "master_raster_{}_{}_{}.tif".format(path, row, year)
    pth = os.path.join(raster_directory, fname)
    mask_fname = "class_mask_{}_{}_{}.tif".format(path, row, year)
    mask_path = os.path.join(raster_directory, mask_fname)
    if os.path.isfile(pth):
        print("Master raster already created for {}_{}_{}.".format(path, row, year))
        if os.path.isfile(mask_path):
            print('Class mask template already created for {}_{}_{}'.format(path, row, year))
            return pth
        else:
            print("Creating class mask template.")
            with rasopen(pth, 'r') as src:
                meta = src.meta.copy()
                h = meta['height']
                w = meta['width']

            meta.update(count=1, dtype=float32)

            with rasopen(mask_path, 'w', **meta) as msk:
                out = zeros((h, w)).astype(float32)
                msk.write(out, 1)
            return pth
        
    first = True
    stack = None
    num_rasters = 0
    for key in paths_map:
        num_rasters += len(paths_map[key])

    j = 0
    for feat in sorted(paths_map.keys()): # ensures the stack is in the same order each time.
        # Ordering within bands is assured by sorting the list that
        # each band corresponding to, as that's sorting by date.
        feature_rasters = paths_map[feat] # maps bands to their location in filesystem.
        for feature_raster in feature_rasters:
            with rasopen(feature_raster, mode='r') as src:
                arr = src.read()
                raster_geo = src.meta.copy()
            if first:
                first_geo = raster_geo.copy()
                empty = zeros((num_rasters, arr.shape[1], arr.shape[2]), float32)
                stack = empty
                stack[j, :, :] = arr
                j += 1
                first = False
            else:
                try:
                    stack[j, :, :] = arr
                    j += 1
                except ValueError: 
                    arr = warp_single_image(feature_raster, first_geo)
                    stack[j, :, :] = arr
                    j += 1

    msk_out = zeros((1, stack.shape[1], stack.shape[2]))
    first_geo.update(count=1, dtype=msk_out.dtype)
    with rasopen(mask_path, mode='w', **first_geo) as msk:
        msk.write(msk_out)

    first_geo.update(count=num_rasters, dtype=stack.dtype)

    with rasopen(pth, mode='w', **first_geo) as dst:
        dst.write(stack)

    print("Master raster saved to {}.".format(pth))

    return pth


def normalize_and_save_image(fname):
    norm = True
    with rasopen(fname, 'r') as rsrc:
        if "normalized" in rsrc.tags():
            return
        else:    
            rass_arr = rsrc.read()
            rass_arr = rass_arr.astype(float32)
            profile = rsrc.profile.copy()
            profile.update(dtype=float32)
            rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
            scaler = StandardScaler() # z-normalization
            scaler.fit(rass_arr)
            rass_arr = scaler.transform(rass_arr)
    with rasopen(fname, 'w', **profile) as dst:
        dst.write(rass_arr, 1)
        print("Normalized", fname)
        dst.update_tags(normalized=True)


def raster_sum(raster):
    with rasopen(raster, 'r') as src:
        arr_masked = src.read(1, masked=True) # get rid of nodata values
    s = arr_masked.sum()
    count = arr_masked.count()
    return s, count


def raster_squared_sum(raster, mean):
    with rasopen(raster, 'r') as src:
        arr_masked = src.read(1, masked=True) # get rid of nodata values
    squared_diff = (arr_masked - mean)**2
    s = squared_diff.sum()
    count = squared_diff.count()
    return s, count


def bandwise_stddev(paths_list, band_name, band_mean):
    ''' Calculate the stddev of the pixel
    values in a given band through time.'''
    n_pixels = 0
    pixel_value_squared_sum = 0
    for filepath in paths_list:
        p_sum, num_pix = raster_squared_sum(filepath, band_mean)
        pixel_value_squared_sum += p_sum
        n_pixels += num_pix
    if n_pixels == 0:
        print("0 non masked pixels.")
        return 1
    return (sqrt(pixel_value_squared_sum / n_pixels), band_name)


def bandwise_mean(paths_list, band_name):
    n_pixels = 0
    pixel_value_sum = 0
    for filepath in paths_list:
        p_sum, num_pix = raster_sum(filepath)
        pixel_value_sum += p_sum
        n_pixels += num_pix
    if n_pixels == 0:
        print("0 non masked pixels.")
        return 1
    return (pixel_value_sum / n_pixels, band_name)


def download_images(project_directory, path, row, year, satellite=8, n_landsat=3, 
        max_cloud_pct=40):

    image_stack = ImageStack(satellite=satellite, path=path, row=row, root=project_directory,
            max_cloud_pct=max_cloud_pct, n_landsat=n_landsat, year=year)

    image_stack.build_evaluating() # the difference b/t build_training() and build_eval() is
    # a cloud mask.
    return image_stack


def get_wrs2_features(path, row):

    with fopen(WRS2) as src:
        for feat in src:
            poly = shape(feat['geometry'])
            propr = feat['properties']
            if propr['PATH'] == path and propr['ROW'] == row:
                return [feat]
    return None


def all_rasters(image_directory, satellite=8):
    ''' Recursively get all rasters in image_directory
    and its subdirectories, and adds them to band_map. '''
    band_map = defaultdict(list)
    for band in landsat_rasters()[satellite]:
        band_map[band] = []
    for band in static_rasters():
        band_map[band] = []
    for band in climate_rasters():
        band_map[band] = []

    extensions = (".tif", ".TIF")
    for dirpath, dirnames, filenames in os.walk(image_directory):
        for f in filenames:
            if any(ext in f for ext in extensions):
                for band in band_map:
                    if f.endswith(band):
                        band_map[band].append(os.path.join(dirpath, f))

    for band in band_map:
        band_map[band] = sorted(band_map[band]) # ensures ordering within bands - sort by time.

    return band_map


def raster_means(image_directory, satellite=8):
    """ Gets all means of all images stored
    in image_directory and its subdirectories. 
    Images end with (.tif, .TIF) 
    Image_directory in a typical case would 
    be project_root/image_data/train/.
    This returns band_map, which is a dict of lists with
    keys band names (B1, B2...) and values lists of
    the locations of the rasters in the filesystem."""

    outfile = os.path.join(image_directory, "mean_mapping.pkl")
    if os.path.isfile(outfile):
        with open(outfile, 'rb') as f:
            mean_mapping = pickle.load(f)
        return mean_mapping

    band_map = all_rasters(image_directory, satellite)
    mean_mapping = {}

    for band in band_map:
        mean, bnd = bandwise_mean(band_map[band], band)
        mean_mapping[band] = mean

    with open(outfile, 'wb') as f:
        pickle.dump(mean_mapping, f)

    return mean_mapping


def raster_stds(image_directory, mean_map, satellite=8):

    outfile = os.path.join(image_directory, "stddev_mapping.pkl")
    if os.path.isfile(outfile):
        with open(outfile, 'rb') as f:
            stddev_mapping = pickle.load(f)
        return stddev_mapping

    band_map = all_rasters(image_directory, satellite) # get all rasters 
    # in the image directory
    stddev_mapping = {}

    for band in band_map.keys():
        std, bnd = bandwise_stddev(band_map[band], band, mean_map[band])
        stddev_mapping[band] = std

    with open(outfile, 'wb') as f:
        pickle.dump(stddev_mapping, f)

    pprint('STDMAP')
    pprint(stddev_mapping)
    print("-------")
    pprint('MEANMAP')
    pprint(mean_map)

    return stddev_mapping


def create_all_master_rasters(image_directory, raster_save_directory, mean_mapping=None, 
        stddev_mapping=None):
    """ Creates a master raster for all images in image_directory. 
    Image directory is assumed to be a top-level directory that contains
    all the path_row directories for test or train (image_data/test/path_row_year*/) 
    Image directory is image_data/test/ in this case."""
    dirs = os.listdir(image_directory)
    for sub_dir in dirs:
        out = os.path.join(image_directory, sub_dir)
        if os.path.isdir(out):
            paths_map = all_rasters(out)
            i = 0
            path = sub_dir[:2]
            row = sub_dir[3:5]
            year = sub_dir[-4:]
            create_master_raster(paths_map, path, row, year, raster_save_directory, mean_mapping,
                    stddev_mapping)


def clip_raster(evaluated, path, row, outfile=None):

    shp = gpd.read_file(WRS2)
    out = shp[shp['PATH'] == path]
    out = out[out['ROW'] == row]

    with rasopen(evaluated, 'r') as src:
        out = out.to_crs(src.crs)
        meta = src.meta.copy()
        features = get_features(out)
        out_image, out_transform = mask(src, shapes=features, crop=True, nodata=nan)

    outfile = evaluated
    save_raster(out_image, outfile, meta)


def save_raster(arr, outfile, meta, count=5):
    meta.update(count=count)
    with rasopen(outfile, 'w', **meta) as dst:
        dst.write(arr)


def load_raster(master_raster):
    with rasopen(master_raster, 'r') as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta


def save_model_info(outfile, args):
    template = '{}={}|'
    with open(outfile, 'a') as f:
        for key in args:
            f.write(template.format(key, args[key]))
        f.write("\n-------------------\n")
    print("wrote run info to {}".format(outfile))

if __name__ == "__main__":
    pass
