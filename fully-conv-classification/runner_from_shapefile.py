import os
import pickle
import time
import pdb
from glob import glob
from pprint import pprint
from numpy import save as nsave
from fiona import open as fopen
from rasterio.errors import RasterioIOError
from collections import defaultdict, OrderedDict
from random import choice
from shapely.geometry import shape
from multiprocessing import Pool

from shapefile_utils import filter_shapefile_overlapping
from data_utils import paths_mappings_single_scene, paths_map_multiple_scenes
from runspec import landsat_rasters, static_rasters, climate_rasters
from data_generators import extract_training_data_v2


def assign_shapefile_class_code(shapefile):
    if 'irrigated' in shapefile and 'unirrigated' not in shapefile:
        return 0
    if 'unirrigated' in shapefile:
        return 1
    if 'uncultivated' in shapefile:
        return 2
    if 'wetlands' in shapefile:
        return 3
    if 'fallow' in shapefile:
        return 4


def assign_shapefile_year(shapefile):
    # get the shapefile name, not the whole path.
    return 2013



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
    
    in_train_shapefile_directory =  '/home/thomas/current_training_montana/train/'
    in_test_shapefile_directory = '/home/thomas/current_training_montana/test/'
    split_out_train_shapefile_directory = 'shapefile_data/train/'
    split_out_test_shapefile_directory = 'shapefile_data/test/'
    image_directory = '/home/thomas/share/image_data/train/'
    training_data_directory = '/home/thomas/share/training_data/train/'
    test_data_directory = '/home/thomas/share/training_data/test/'
    fs = [f for f in glob(in_train_shapefile_directory + "*.shp")]
    tf = [split_out_train_shapefile_directory] * len(fs)

    # with Pool() as pool:
    #     pool.starmap(filter_shapefile_overlapping, zip(fs, tf))
    extract_training_data_v2(split_out_train_shapefile_directory, image_directory,
            training_data_directory, assign_shapefile_year, assign_shapefile_class_code, n_classes=6) 
    extract_training_data_v2(split_out_test_shapefile_directory, image_directory,
            test_data_directory, assign_shapefile_year, assign_shapefile_class_code, n_classes=6) 
    # directories = os.listdir(image_directory)
    # test = image_directory + directories[0]
    # scene_dates_and_mappings = paths_mappings_single_scene(test)
    # from pprint import pprint
    # for s in scene_dates_and_mappings:
    #     print(s)
    #     pprint(scene_dates_and_mappings[s])
