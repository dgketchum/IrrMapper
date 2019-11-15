import os
import geopandas as gpd
import json
import pdb
import datetime
import numpy as np

from fiona import open as fopen
from glob import glob
from lxml import html 
from requests import get
from copy import deepcopy
from shapely.geometry import shape, mapping
from collections import defaultdict
from rasterio import float32, open as rasopen
from shapely.geometry import shape, Polygon, mapping
from rasterio.mask import mask
from pickle import load
from multiprocessing import Pool
from sat_image.image import Landsat8

from prepare_images import ImageStack
from crop_data_layer import CropDataLayer as Cdl
from shapefile_utils import get_features
from sat_image.warped_vrt import warp_single_image
from runspec import landsat_rasters, static_rasters, climate_rasters

WRS2 = '../spatial_data/wrs2_descending_usa.shp'

def download_cdl_over_path_row(path, row, year, image_directory):

    out_dir = os.path.join(image_directory, '_'.join([str(path), str(row), str(year)]))
    cdl_mask = os.path.join(out_dir, "cdl_mask.tif")
    if os.path.isfile(cdl_mask):
        print("cdl already downloaded for {} {} {}".format(path, row, year))
        return

    sub_dirs = os.listdir(out_dir)
    if not len(sub_dirs):
        raise ValueError("images not downloaded for {} {} {}".format(path, row, year))

    print(path, row, year)
    for r in sub_dirs:
        if os.path.isdir(os.path.join(out_dir, r)):
            if 'climate' not in r:
                random_landsat_dir = os.path.join(out_dir, r)
                break

    landsat = glob(os.path.join(random_landsat_dir, "*TIF"))[0]
    landsat_pic = landsat
    landsat = Landsat8(random_landsat_dir)
    try:
        polygon = landsat.get_tile_geometry()
        cdl = Cdl(year=year, target_profile=landsat.profile)
        cdl.get_mask(clip_geometry=polygon, out_file=cdl_mask)
    except Exception as e:
        print(e.args)
        print(landsat_pic)




def download_images_over_shapefile(shapefile, image_directory, year):
    '''Downloads p/r corresponding to the location of 
       the shapefile. Image_directory: where to save the raw images.
       '''
    p, r = get_shapefile_path_row(shapefile) 
    suff = str(p) + '_' + str(r) + "_" + str(year)
    landsat_dir = os.path.join(image_directory, suff)
    satellite = 8
    if year < 2013:
        satellite = 7
    if not os.path.isdir(landsat_dir):
        os.mkdir(landsat_dir)
        ims = _download_images(landsat_dir, p, r, year, satellite)
    else:
        ims = _download_images(landsat_dir, p, r, year, satellite)

    return ims


def download_from_pr(p, r, year, image_directory, satellite=8):
    '''Downloads p/r corresponding to the location of 
       the shapefile.'''
    # TODO: add rasterioIOError error checking
    # and resolution here.
    suff = str(p) + '_' + str(r) + "_" + str(year)
    landsat_dir = os.path.join(image_directory, suff)
    if not os.path.isdir(landsat_dir):
        os.mkdir(landsat_dir)
    ims = _download_images(landsat_dir, p, r, year, satellite)
    return ims


def _download_images(project_directory, path, row, year, satellite,
        n_landsat=3, max_cloud_pct=40):

    image_stack = ImageStack(satellite=satellite, path=path, row=row, root=project_directory,
            max_cloud_pct=max_cloud_pct, n_landsat=n_landsat, year=year)

    image_stack.build_evaluating() # the difference b/t build_training() and build_eval() is
    # a cloud mask.
    return image_stack


def _parse_landsat_capture_date(landsat_scene):
    '''
    returns: calendar date of scene capture
    landsat_scene is a directory (i.e data/38_27_2013/')
    scene ID:
       LXSPPPRRRYYYYDDDGSIVV
       L = Landsat
       X = Sensor
       S = Satellite
       PPP = WRS Path
       RRR = WRS Row
       YYYY = Year
       DDD = Julian day
       GSI = Ground station ident
       VV = Archived version number
    '''
    julian_year_day = landsat_scene[-10:-5]
    return datetime.datetime.strptime(julian_year_day, '%y%j').date()


def _landsat_band_map(subdirectory, satellite=8):
    band_map = dict()

    for band in landsat_rasters()[satellite]:
        band_map[band] = None
    for band in static_rasters():
        band_map[band] = None
    for band in climate_rasters():
        band_map[band] = None

    extensions = (".tif", ".TIF")
    for dirpath, dirnames, filenames in os.walk(subdirectory):
        for f in filenames:
            if any(ext in f for ext in extensions):
                for band in band_map:
                    if f.endswith(band):
                        band_map[band] = os.path.join(dirpath, f)
    return band_map


def _climate_band_map(directory, band_map, date):

    files = glob(os.path.join(directory, '*.tif'))
    files.extend(glob(os.path.join(directory, '*.TIF')))
    for f in files:
        datestring = os.path.basename(f)[:10]
        cur = datetime.datetime.strptime(datestring, '%Y-%m-%d').date()
        if date == cur:
            for band in band_map:
                if f.endswith(band):
                    band_map[band] = os.path.join(directory, f)
    return band_map


def paths_mapping_single_scene(landsat_directory):
    directories = [os.path.join(landsat_directory, f) for f in os.listdir(landsat_directory) if
            os.path.isdir(os.path.join(landsat_directory, f))]
    climate_directory  = os.path.join(landsat_directory, 'climate_rasters')
    other_rasters = [os.path.join(landsat_directory, f) for f in os.listdir(landsat_directory) if
            not os.path.isdir(os.path.join(landsat_directory, f))]
    date_dict = dict()
    for d in directories:
        if 'climate' in d:
            continue
        pm = _landsat_band_map(d)
        date = _parse_landsat_capture_date(d)
        cm = _climate_band_map(climate_directory, pm, date)
        for raster in other_rasters:
            for band in static_rasters():
                if raster.endswith(band):
                    pm[band] = raster
        date_dict[date] = pm 
    return date_dict


def paths_map_multiple_scenes(image_directory, satellite=8):
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


def mean_of_three(paths_map, image_stack, target_shape, satellite=8):

    # iterate over paths_map
    # iterate over each raster in paths_map

    j = 0
    out_image_stack = np.zeros((19, target_shape[1], target_shape[2]))
    out_idx = 0
    for band in sorted(paths_map.keys()):
        if band in landsat_rasters()[satellite]:
            for sub_band in paths_map[band]:
                out_image_stack[out_idx] += image_stack[j]
                j += 1
            out_image_stack[out_idx] /= 3
            out_idx += 1
        else:
            out_image_stack[out_idx] = image_stack[j]
            out_idx += 1

    return out_image_stack


def median_of_three(paths_map, image_stack, target_shape, satellite=8):

    j = 0
    out_image_stack = np.zeros((19, target_shape[1], target_shape[2]))
    out_idx = 0
    for band in sorted(paths_map.keys()):
        if band in landsat_rasters()[satellite]:
            slc = np.zeros((3, target_shape[1], target_shape[2]))
            for i, sub_band in enumerate(paths_map[band]):
                slc[i] = image_stack[j]
                j += 1
            out_image_stack[out_idx] = np.median(slc, axis=0)
            out_idx += 1
        else:
            out_image_stack[out_idx] = image_stack[j]
            out_idx += 1

    return out_image_stack


def map_bands_to_indices(target_bands, satellite=8):

    band_map = defaultdict(list)
    for band in landsat_rasters()[satellite]:
        band_map[band] = []
    for band in static_rasters():
        band_map[band] = []
    for band in climate_rasters():
        band_map[band] = []

    image_directory = '/home/thomas/share/image_data/train/37_28_2013/'
    extensions = (".tif", ".TIF")
    for dirpath, dirnames, filenames in os.walk(image_directory):
        for f in filenames:
            if any(ext in f for ext in extensions):
                for band in band_map:
                    if f.endswith(band):
                        band_map[band].append(os.path.join(dirpath, f))

    for band in band_map:
        band_map[band] = sorted(band_map[band]) # ensures ordering within bands - sort by time.

    indices = []
    i = 0
    for feat in sorted(band_map.keys()): # ensures the stack is in the same order each time.
        feature_rasters = band_map[feat]
        for feature_raster in feature_rasters:
            for band in target_bands:
                if feature_raster.endswith(band):
                    indices.append(i)
                    i += 1
    return indices


def _maybe_warp(feature_raster, target_geo, target_shape):
    arr, _ = load_raster(feature_raster)
    if not arr.shape == target_shape:
        arr = warp_single_image(feature_raster, target_geo)
    return arr, feature_raster


def _load_rasters(paths_map, target_geo, target_shape):
    single_band = False
    num_rasters = 0
    for key in paths_map:
        if isinstance(paths_map[key], str):
            single_band = True
            num_rasters += 1
        else:
            num_rasters += len(paths_map[key])
    j = 0

    if not single_band:
        feature_rasters = [feature_raster for feat in paths_map.keys() for feature_raster in
        paths_map[feat]]
    else:
        feature_rasters = [paths_map[feat] for feat in paths_map.keys()]
    tg = [target_geo]*len(feature_rasters)
    ts = [target_shape]*len(feature_rasters)
    if not single_band:
        feature_rasters = [feature_raster for feat in paths_map.keys() for feature_raster in
        paths_map[feat]]
    else:
        feature_rasters = [paths_map[feat] for feat in paths_map.keys()]
    tg = [target_geo]*len(feature_rasters)
    ts = [target_shape]*len(feature_rasters)
    with Pool() as pool:
        # Multiprocess the loading of rasters into memory.
        # Speedup of ~40s.
        out = pool.starmap(_maybe_warp, zip(feature_rasters, tg, ts))
    rasters = {feature_raster: array for (array, feature_raster) in out}
    return rasters, num_rasters


def stack_rasters_multiprocess(paths_map, target_geo, target_shape):
    first = True
    stack = None
    single_band = False
    j = 0
    rasters, num_rasters = _load_rasters(paths_map, target_geo, target_shape)
    for feat in sorted(paths_map.keys()): # ensures the stack is in the same order each time.
        # Ordering within bands is assured by sorting the list that
        # each band corresponding to, as that's sorted by date.
        feature_rasters = paths_map[feat] # maps bands to their location in filesystem.
        if not single_band:
            for feature_raster in feature_rasters:
                arr = rasters[feature_raster]
                if first:
                    stack = np.zeros((num_rasters, target_shape[1], target_shape[2]), np.uint16)
                    stack[j, :, :] = arr
                    j += 1
                    first = False
                else:
                    stack[j, :, :] = arr
                    j += 1 
        else:

            arr, _ = _select_correct_band(rasters, feat)

            # somehow select
            if first:
                stack = np.zeros((num_rasters, target_shape[1], target_shape[2]), np.uint16)
                stack[j, :, :] = arr
                j += 1
                first = False
            else:
                stack[j, :, :] = arr
                j += 1 
    return stack


def _select_correct_band(rasters, target_feat):
    for path in rasters:
        if path.endswith(target_feat):
            return rasters[path], path


def stack_rasters(paths_map, target_geo, target_shape):
    first = True
    stack = None
    num_rasters = 0
    for key in paths_map: num_rasters += len(paths_map[key])
    j = 0
    for feat in sorted(paths_map.keys()): # ensures the stack is in the same order each time.
        # Ordering within bands is assured by sorting the list that
        # each band corresponding to, as that's sorting by date.
        feature_rasters = paths_map[feat] # maps bands to their location in filesystem.
        for feature_raster in feature_rasters:
            with rasopen(feature_raster, mode='r') as src:
                arr = src.read()
            if first:
                stack = np.zeros((num_rasters, target_shape[1], target_shape[2]), np.uint16)
                stack[j, :, :] = arr
                j += 1
                first = False
            else:
                try:
                    stack[j, :, :] = arr
                    j += 1
                except ValueError: 
                    arr = warp_single_image(feature_raster, target_geo)
                    stack[j, :, :] = arr
                    j += 1
    return stack


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


def _get_path_row_geometry(path, row):
    shp = gpd.read_file(WRS2)
    out = shp[shp['PATH'] == int(path)]
    out = out[out['ROW'] == int(row)]
    return out


def clip_raster(evaluated, path, row, outfile=None):

    out = _get_path_row_geometry(path, row)

    with rasopen(evaluated, 'r') as src:
        out = out.to_crs(src.crs['init'])
        features = get_features(out)
        # if crop == true for mask, you have to update the metadata.
        out_image, out_transform = mask(src, shapes=features, crop=True, nodata=np.nan)
        meta = src.meta.copy()
        count = out_image.shape[0]

    meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})
    if outfile is not None:
        save_raster(out_image, outfile, meta, count)


def save_raster(arr, outfile, meta, count=5):
    meta.update(count=count)
    with rasopen(outfile, 'w', **meta) as dst:
        dst.write(arr)


def load_raster(raster_name):
    with rasopen(raster_name, 'r') as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta

prs =  [[34, 26], [36, 27], [37,28], [34,27], [39,26], [37,29], [42,27], [41,28], [39,29], [36,28], [40,27], [37,26], [35,26], [38,29], [40,28], [38,27], [35,27], [42,26], [41,26], [40,29], [34,29], [35,29], [38,26], [36,26], [39,28], [41,27], [38,28], [37,27], [36,29], [35,28], [43,26], [39,27], [40,26], [43,27], [34,28]]

if __name__ == "__main__":

    from runspec import landsat_rasters, climate_rasters

    year = 2013
    for path, row in prs:
         # download_from_pr(int(path), int(row), int(year), '/home/thomas/share/image_data/')
         download_cdl_over_path_row(path, row, year, '/home/thomas/share/image_data/')

