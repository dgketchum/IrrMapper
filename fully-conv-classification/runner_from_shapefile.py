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
    image_test_directory = '/home/thomas/share/image_data/test'

    image_dirs = [image_train_directory, image_test_directory]
    shp_train = 'shapefile_data/train/'
    shp_test = 'shapefile_data//test/'
    shp_dirs = [shp_train, shp_test]
    master_train = '/home/thomas/share/master_rasters/train/'
    master_test = '/home/thomas/share/master_rasters/test'
    master_dirs = [master_train, master_test]
    year = 2013
    # for s, i in zip(shp_dirs, image_dirs):
    #     download_all_images(i, s, year)
    # for im_dir, mas_dir in zip(image_dirs, master_dirs):
    #     create_all_master_rasters(im_dir, mas_dir) 
    master_train = '/home/thomas/share/master_rasters/train/'
    master_test = '/home/thomas/share/master_rasters/test/'
    image_train = '/home/thomas/share/image_data/train/'
    image_test = '/home/thomas/share/image_data/test/' 
    irr1 = 'Huntley'
    irr2 = 'Sun_River'
    fallow = 'Fallow'
    forest = 'Forrest'
    other = 'other'
    target_dict = {irr2:0, irr1:0, fallow:1, forest:2, other:3}
    augment_dict = {0:True, 1:False, 2:False, 3:True}
    train_dir = 'training_data/train/'
    shp_train = 'shapefile_data/train/'
    save = True
    pixel_dict = extract_training_data(target_dict, shp_train, image_train,
            master_train, train_dir, save=save, augment_dict=augment_dict) 
    print("{} instances in each class.".format(pixel_dict))
    max_weight = max(pixel_dict.values())
    for key in pixel_dict:
        print(key, max_weight / pixel_dict[key])
    tot = 0
    test_dir = 'training_data/test/'
    shp_test = 'shapefile_data/test/'
    pixel_dict = extract_training_data(target_dict, shp_test, image_test, master_test, 
            test_dir, save=save, augment_dict=augment_dict)
    print("And {} instances in each class.".format(pixel_dict))
