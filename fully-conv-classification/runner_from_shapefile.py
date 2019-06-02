import os
import pickle
from glob import glob
from pprint import pprint
import time
from numpy import save as nsave
from fiona import open as fopen
from collections import defaultdict, OrderedDict
from random import choice
from shapely.geometry import shape
from data_utils import download_images
from shapefile_utils import filter_shapefile_overlapping
from runspec import landsat_rasters, static_rasters, climate_rasters
from data_generators import extract_training_data


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
        ims = download_images(landsat_dir, p, r, year, satellite)
    else:
        ims = download_images(landsat_dir, p, r, year, satellite)

    return ims


def download_from_pr(p, r, year, image_directory):
    '''Downloads p/r corresponding to the location of 
       the shapefile, and creates master raster'''
    suff = str(p) + '_' + str(r) + "_" + str(year)
    landsat_dir = os.path.join(image_directory, suff)
    satellite = 8
    if year < 2013:
        satellite = 7
    if not os.path.isdir(landsat_dir):
        os.mkdir(landsat_dir)
        ims = download_images(landsat_dir, p, r, year, satellite)
    else:
        ims = download_images(landsat_dir, p, r, year, satellite)
    return ims


def assign_shapefile_class_code(shapefile):
    if 'irrigated' in shapefile and 'unirrigated' not in shapefile:
        return 0
    if 'unirrigated' in shapefile:
        return 1
    if 'uncultivated' in shapefile:
        return 2
    if 'wetlands' in shapefile:
        return 3


def assign_shapefile_year(shapefile):
    # get the shapefile name, not the whole path.
    shapefile = os.path.basename(shapefile)
    if '2013' in shapefile:
        return 2013
    if '2014' in shapefile:
        return 2014
    if '2015' in shapefile:
        return 2015
    if '2016' in shapefile:
        return 2016
    if '2017' in shapefile:
        return 2017
    return 2014



if __name__ == "__main__":
    # out_shapefile_directory = 'shapefile_data'
    # shp = "/home/thomas/IrrigationGIS/western_states_irrgis/MT/MT_Main/" 
    # This project is becoming more complicated.
    # Needs a test / train organization
    # 1. Filter shapefiles. Can I fix this? Yes. Online splitting.
    # 2. Download images over shapefiles
    # 3. Extract training data
    # 4. Train network.

    # Need to download images.

    shapefile_directory = '/home/thomas/IrrigationGIS/western_states_irrgis/reprojected_western_gis/post-2013'
    image_directory = '/home/thomas/share/image_data/train/'
    shapefiles = [f for f in glob(os.path.join(shapefile_directory, "*.shp"))]
    for f in shapefiles:
        print("Downloading images for {}".format(f))
        path_row_map = filter_shapefile_overlapping(f)
        year = assign_shapefile_year(f)
        for path_row in path_row_map:
            path = int(path_row[0:2])
            row = int(path_row[-2:])
            try:
                print(path, row, year)
                download_from_pr(path, row, year, image_directory)
            except Exception as e:
                print(e)
                time.sleep(3)
