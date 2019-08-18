import os
import pickle
import time
import pdb
from glob import glob
from pprint import pprint
from collections import defaultdict, OrderedDict
from multiprocessing import Pool

from shapefile_utils import filter_shapefile_overlapping
from data_utils import paths_mappings_single_scene, paths_map_multiple_scenes
from extract_training_data import extract_training_data_multiple_classes_per_instance


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


# fs = [f for f in glob(in_test_shapefile_directory + "*.shp")]
# tf = [split_out_test_shapefile_directory] * len(fs)
# with Pool() as pool:
#     pool.starmap(filter_shapefile_overlapping, zip(fs, tf))
# fs = [f for f in glob(in_train_shapefile_directory + "*.shp")]
# tf = [split_out_train_shapefile_directory] * len(fs)
# with Pool() as pool:
#     pool.starmap(filter_shapefile_overlapping, zip(fs, tf))

if __name__ == "__main__":
    # out_shapefile_directory = 'shapefile_data'
    # shp = "/home/thomas/IrrigationGIS/western_states_irrgis/MT/MT_Main/" 
    # This project is becoming more complicated.
    # Needs a test / train organization
    # 1. Filter shapefiles. 
    # 2. Download images over shapefiles
    # 3. Extract training data
    #    - with offline augmentation.  
    # 4. Train network.
    # Need to download images.
    
    in_train_shapefile_directory =  '/home/thomas/current_training_montana/train/'
    in_test_shapefile_directory = '/home/thomas/current_training_montana/test/'
    split_out_train_shapefile_directory = 'shapefile_data/train/'
    split_out_test_shapefile_directory = 'shapefile_data/test/'
    image_directory = '/home/thomas/share/image_data/train/'
    training_data_directory = '/home/thomas/ssd/training_data/train_mc/'
    test_data_directory = '/home/thomas/ssd/training_data/test_mc/'
    offline_augmentation_dict = {0:200, 1:608, 2:608, 3:400, 4:200, 5:608}
    extract_training_data_multiple_classes_per_instance(split_out_train_shapefile_directory,
            image_directory, training_data_directory, assign_shapefile_year,
            assign_shapefile_class_code, n_classes=6) 
    offline_augmentation_dict = {0:608, 1:608, 2:608, 3:608, 4:608, 5:608}
    extract_training_data_multiple_classes_per_instance(split_out_test_shapefile_directory,
            image_directory, test_data_directory, assign_shapefile_year,
            assign_shapefile_class_code, n_classes=6) 
