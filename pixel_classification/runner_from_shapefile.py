import warnings
import glob
import os
import gc
from  multiprocessing import Pool
from numpy import save as nsave
from compose_array_single_shapefile import PTASingleShapefile
from fiona import open as fopen
from shapely.geometry import shape
from data_utils import download_images, get_shapefile_path_row, split_shapefile, create_master_raster, create_master_masked_raster

def create_training_data(shapefile, shapefile_directory, image_directory, class_code,
        kernel_size, instances, training_directory, year, raster_directory, chunk_size=2000, save=True):

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

    ms = create_master_raster(ims, p, r, year, raster_directory)
    mms =  create_master_masked_raster(ims, p, r, year, raster_directory)

    shp_path = os.path.join(shapefile_directory, shapefile)
    pta = PTASingleShapefile(shapefile_path=shp_path, master_raster=ms,
            training_directory=training_directory, overwrite_points=False, class_code=class_code,
            path=p, row=r, paths_map=ims.paths_map, masks=ims.masks,
            instances=instances, kernel_size=kernel_size, sz=chunk_size)

    pta.extract_sample()

def get_all_shapefiles(to_match, year, data_directory, irrigated):
    ''' Get all shapefiles in same p/r as to_match '''
    p, r = get_shapefile_path_row(os.path.join(data_directory, to_match))
    ls = []
    for f in glob.glob(data_directory + "*.shp"):
        if "sample_points" not in f:
            pp, rr = get_shapefile_path_row(f)
            if pp == p and rr == r:
                oup = False
                for key in irrigated:
                    if key in f:
                        oup = True
                if not oup:
                    ls.append(f)
    return ls

def shapefile_area(shapefile):
    summ = 0
    with fopen(shapefile, "r") as src:
        for feat in src:
            poly = shape(feat['geometry'])
            summ += poly.area
    return summ

def get_total_area(data_directory, filenames):
    ''' Gets the total area of the polygons
        in the files in filenames '''
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

def extract_data(data_directory, names, n_instances, class_code, kernel_size):

    def is_it(f, names):
        for e in names:
            if e in f:
                return True
        return False

    total_area = get_total_area(data_directory, names) # units?
    for f in glob.glob(data_directory + "*.shp"):
        if is_it(f, names) and 'sample' not in f:
            req_points = required_points(f, total_area, n_instances)
            ff = os.path.basename(f)
            create_training_data(ff, data_directory, image_directory,
                    class_code, kernel_size, req_points, train_dir, 2013, raster_dir) 
        gc.collect()

def go(f):
    data_directory = 'split_shapefiles_west/'
    shp_dir = '/home/thomas/IrrigationGIS/western_states_irrgis/western_gis_backup'
    fname = os.path.basename(f) 
    split_shapefile(shp_dir, fname, data_directory)


if __name__ == "__main__":

    irrigated = ['MT_Sun_River_2013', "MT_Huntley_Main_2013"]
    other = ['other']
    fallow = ['Fallow']
    forest = ['Forrest']

    train_dir = 'training_data/'
    data_directory = 'shapefile_data_western_us/'
    image_directory = 'image_data/'
    raster_dir = 'master_rasters'
    kernel_size = 57


    shp_dir = '/home/thomas/IrrigationGIS/western_states_irrgis/western_gis_backup/'
    fnames = [f for f in glob.glob(shp_dir + "*.shp") if 'reproj' in f]
    instances = [50000, 1e5, 1e5, 1e5]
    i2 = [1e5, 1e5]
    class_code = [0, 1, 2, 3]
    c2 = [2, 3]
    dd = [data_directory]*2
    dd2 = dd.copy()
    ks = [41]*2
    ks2 = ks.copy()
    names = [irrigated, other] 
    names2 = [fallow, forest]
    # note: the extraction of training data took 6h 29m
    # extract_data(dd[0], fallow, 1e5, 2, 41)
    with Pool() as pool:
        pool.starmap(extract_data, zip(dd, names, instances, class_code, ks))
    with Pool() as pool:
        pool.starmap(extract_data, zip(dd2, names2, i2, c2, ks2))

    #with Pool(os.cpu_count()) as pool:
    #    pool.map(go, fnames)
     # 12 minutes to 5 and a half.


