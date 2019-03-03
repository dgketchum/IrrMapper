import warnings
import os
import glob
import pickle
from  multiprocessing import Pool
from numpy import save as nsave
from compose_array_single_shapefile import PTASingleShapefile, ShapefileSamplePoints
from fiona import open as fopen
from shapely.geometry import shape
from data_utils import download_images, get_shapefile_path_row, split_shapefile, create_master_raster, filter_shapefile


def download_images_over_shapefile(shapefile, image_directory, year, master_raster_directory):
    '''Downloads p/r corresponding to the location of 
       the shapefile, and creates master raster.
       Image_directory: where to save the raw images.
       mr_directory: "                    " master_rasters.'''
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

    ms = create_master_raster(ims, p, r, year, master_raster_directory)
    
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

    ms = create_master_raster(ims, p, r, year, master_raster_directory)
    
    return ims

def sample_points_from_shapefile(shapefile_path, instances):
    ssp = ShapefileSamplePoints(shapefile_path, m_instances=instances)
    ssp.create_sample_points(save_points=True)
    return ssp.outfile 

def shapefile_area(shapefile):
    summ = 0
    with fopen(shapefile, "r") as src:
        for feat in src:
            poly = shape(feat['geometry'])
            summ += poly.area
    return summ

def get_total_area(data_directory, filenames):
    ''' Gets the total area of the polygons
        in the files in filenames
        TODO: Get an equal-area projection'''

    tot = 0
    for f in glob.glob(data_directory + "*.shp"):
        if "sample" not in f:
            for f2 in filenames:
                if f2 in f:
                    tot += shapefile_area(f)
    return tot

def required_points(shapefile, total_area, total_instances):
    area = shapefile_area(shapefile)
    frac = area / total_area
    return int(total_instances * frac)

def split_shapefiles_multiproc(f):
    data_directory = 'split_shapefiles_west/'
    shp_dir = '/home/thomas/IrrigationGIS/western_states_irrgis/western_gis_backup'
    fname = os.path.basename(f) 
    split_shapefile(shp_dir, fname, data_directory)

def download_all_images(image_directory, shapefile_directory, year=2013):
    ''' Downloads all images over each shapefile in
    shapefile directory '''
    template = "{}_{}_{}"
    done = set()
    satellite = 8
    all_paths = []
    for f in glob.glob(os.path.join(shapefile_directory, "*.shp")):
        p, r = get_shapefile_path_row(f)
        t = template.format(p, r, year)
        if t not in done:
            done.add(t)
            ims = download_images_over_shapefile(f, image_directory, year, master)
            all_paths.append(ims.paths_map)

    with open("path_map.pkl", 'wb') as f:
        pickle.dump(all_paths, f)

    return all_paths


if __name__ == "__main__":
    # out_shapefile_directory = 'shapefile_data'
    # shp = "/home/thomas/IrrigationGIS/western_states_irrgis/MT/MT_Main/" 
    # for f in glob.glob(shp + "*.shp"):
    #     filter_shapefile(f, out_shapefile_directory)

    image_directory = 'image_data/'
    shapefile_directory = 'shapefile_data/all_shapefiles'
    master = 'master_rasters/'
    year = 2013
    all_paths = download_all_images(image_directory, shapefile_directory, year)
