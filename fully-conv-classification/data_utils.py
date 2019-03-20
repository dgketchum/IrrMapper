import glob
import os
import geopandas as gpd
import json
from fiona import open as fopen
from lxml import html
from requests import get
from copy import deepcopy
from numpy import zeros, asarray, array, reshape, nan, sqrt, std
from shapely.geometry import shape
from collections import defaultdict
from rasterio import float32, open as rasopen
from rasterio.mask import mask
from prepare_images import ImageStack
from sat_image.warped_vrt import warp_single_image


def create_master_raster(paths_map, path, row, year, raster_directory, mean_map, stddev_map):
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
            band_mean = None
            for band in mean_map:
                if feature_raster.endswith(band):
                    band_mean = mean_map[band]
                    band_std = stddev_map[band]

            if band_mean is None:
                print("Band mean not found in mean_mapping for {}".format(feature_raster))
                return

            with rasopen(feature_raster, mode='r') as src:
                arr = src.read()
                raster_geo = src.meta.copy()

            arr = (arr - band_mean) / band_std

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
                    # error can be thrown here if source raster doesn't have crs
                    # OR ! Because rasterio version.
                    # However, deepcopy becomes an issue with the latest
                    # version of rasterio.
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
    ''' Need to center the data to have 
    a zero mean. This means iterating over all images, 
    and taking the "band-wise" mean, then subtracting
    that mean from the band. This mean should
    also only be computed for the test set, but applied
    to the training set. ''' 
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


def download_images(project_directory, path, row, year, satellite=8, n_landsat=3):

    image_stack = ImageStack(satellite=satellite, path=path, row=row, root=project_directory,
            max_cloud_pct=70, n_landsat=n_landsat, year=year)

    image_stack.build_evaluating() # the difference b/t build_training() and build_eval() is
    # a cloud mask.
    return image_stack


def clip_rasters(evaluated_tif_dir, include_string):
    for f in glob(os.path.join(evaluated_tif_dir, "*.tif")):
        if include_string in f:
            out = os.path.basename(f)
            out = out[out.find("_")+1:]
            out = out[out.find("_")+1:]
            out = out[out.find("_")+1:]
            path = out[:2]
            row = out[3:5]
            clip_raster(f, int(path), int(row), outfile=f)


def clip_raster(evaluated, path, row, outfile=None):

    shp = gpd.read_file(WRS2)

    with rasopen(evaluated, 'r') as src:
        shp = shp.to_crs(src.crs)
        meta = src.meta.copy()
        features = get_features(shp, path, row)
        out_image, out_transform = mask(src, shapes=features, nodata=np.nan)

    if outfile:
        save_raster(out_image, outfile, meta)


def save_raster(arr, outfile, meta, count=4):
    meta.update(count=count+1)
    with rasopen(outfile, 'w', **meta) as dst:
        dst.write(arr)


def load_raster(master_raster):
    with rasopen(master_raster, 'r') as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta


if __name__ == "__main__":
    pass
