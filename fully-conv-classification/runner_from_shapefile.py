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
from runspec import assign_shapefile_class_code, assign_shapefile_year


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
    training_data_directory = '/home/thomas/ssd/training_data/four_class/train/'
    test_data_directory = '/home/thomas/ssd/training_data/four_class/test/'
    # assign_shapefile_class_code = assign_shapefile_class_code_binary
    n_classes = 5
    offline_augmentation_dict = {0:200, 1:608, 2:608, 3:400, 4:200, 5:608}
    # todo: add multiple directories to this function.
    # that would probably reduce runtime to half.
    extract_training_data_multiple_classes_per_instance(split_out_train_shapefile_directory,
            image_directory, training_data_directory, assign_shapefile_year,
            assign_shapefile_class_code, n_classes=n_classes) 
    offline_augmentation_dict = {0:608, 1:608, 2:608, 3:608, 4:608, 5:608}
    extract_training_data_multiple_classes_per_instance(split_out_test_shapefile_directory,
            image_directory, test_data_directory, assign_shapefile_year,
            assign_shapefile_class_code, n_classes=n_classes) 
