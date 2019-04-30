import os
import pickle
from glob import glob
from pprint import pprint
from numpy import save as nsave
from fiona import open as fopen
from collections import defaultdict, OrderedDict
from shapely.geometry import shape
from data_utils import download_images, create_master_raster, bandwise_mean, bandwise_stddev
from shapefile_utils import get_shapefile_path_row, split_shapefile, filter_shapefile
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


def download_from_pr(p, r, image_directory, year, master_raster_directory):
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


def download_all_images(image_directory, shapefile_directory, year=2013):
    ''' Downloads all images over each shapefile in
    shapefile directory, and places them in image_directory.'''
    template = "{}_{}_{}"
    done = set()
    satellite = 8
    all_paths = []
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        p, r = get_shapefile_path_row(f)
        t = template.format(p, r, year)
        if t not in done:
            done.add(t)
            ims = download_images_over_shapefile(f, image_directory, year)
    print("Done downloading images for {}. Make sure there were no 503 codes returned".format(shapefile_directory))


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
    if '2013' in shapefile:
        return 2013
    if '2014' in shapefile:
        return 2014
    if '2015' in shapefile:
        return 2015
    if '2016' in shapefile:
        return 2016
    return 2014


if __name__ == "__main__":
    # out_shapefile_directory = 'shapefile_data'
    # shp = "/home/thomas/IrrigationGIS/western_states_irrgis/MT/MT_Main/" 
    # This project is becoming more complicated.
    # Needs a test / train organization
    # 1. Filter shapefiles.
    # 2. Download images over shapefiles
    # 3. Get all means/stddevs
    # 4. Create master rasters
    # 5. Extract training data
    # 6. Train network.

    image_train_directory = '/home/thomas/share/image_data/train/'
    image_test_directory = '/home/thomas/share/image_data/test/'
    image_dirs = [image_train_directory, image_test_directory]
    shp_train = 'shapefile_data/train/'
    shp_test = 'shapefile_data/test/'
    shp_dirs = [shp_train, shp_test]
    master_train = '/home/thomas/share/master_rasters/train/'
    master_test = '/home/thomas/share/master_rasters/test'
    master_dirs = [master_train, master_test]
    shapefile_directory = 'shapefile_data/western_us/split_shapefiles/clipped_to_target_states/'
    i = 0
    '''
    fs = [f for f in glob(shapefile_directory + "*.shp")] 
    print(len(fs))
    for f in glob(shapefile_directory + "*.shp"):
        if '2013' in f:
            year = 2013
        elif '2014' in f:
            year = 2014
        elif '2015' in f:
            year = 2015
        elif '2016' in f:
            year = 2016
        else:
            year = 2014
        download_images_over_shapefile(f, image_train_directory, year)
        print("{} of {} done.".format(i+1, len(fs)))
        i += 1
    '''
    image_train = '/home/thomas/share/image_data/train/'
    image_test = '/home/thomas/share/image_data/train/' 
    train_dir = '/home/thomas/share/training_data/train/'
    shp_train = 'shapefile_data/train/'
    pixel_dict = extract_training_data(shp_train, image_train,
            train_dir, assign_shapefile_year=assign_shapefile_year,
            assign_shapefile_class_code=assign_shapefile_class_code) 
    print("{} instances in each class.".format(pixel_dict))
    max_weight = max(pixel_dict.values())
    for key in pixel_dict:
        print(key, max_weight / pixel_dict[key])
    tot = 0
    test_dir = '/home/thomas/share/training_data/test/'
    shp_test = 'shapefile_data/test/'
    pixel_dict = extract_training_data(shp_test, image_test,
            test_dir, assign_shapefile_year=assign_shapefile_year,
            assign_shapefile_class_code=assign_shapefile_class_code) 
    print("And {} instances in each class.".format(pixel_dict))
