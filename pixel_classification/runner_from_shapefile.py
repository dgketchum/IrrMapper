import warnings
import glob
import os
from numpy import save as nsave
from compose_array_single_shapefile import PTASingleShapefile
from data_utils import download_images, get_shapefile_path_row, split_shapefile, create_master_raster, create_master_masked_raster

def create_training_data(shapefile, shapefile_directory, image_directory, class_code,
        kernel_size, instances, training_directory, year, raster_directory, save=True):


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
        print("Images may have been downloaded for {}_{}_{}".format(p, r, year))
        print("Check to make sure they're all there.")
        ims = download_images(landsat_dir, p, r, year, satellite)
    
    ms = create_master_raster(ims, p, r, year, raster_directory)
    mms =  create_master_masked_raster(ims, p, r, year, raster_directory)

    shp_path = os.path.join(shapefile_directory, shapefile)
    pta = PTASingleShapefile(shapefile_path=shp_path, master_raster=ms,
            training_directory=training_directory, overwrite_points=True, class_code=class_code,
            path=p, row=r, paths_map=ims.paths_map, masks=ims.masks, 
            instances=instances, kernel_size=kernel_size)

    pta.extract_sample()

irrigated = {'MT_Sun_River_2013':2013,
        "MT_Huntley_Main_2013":2013}


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


if __name__ == "__main__":
    
    shp_dir = '/home/thomas/IrrigationGIS/western_states_irrgis/MT/MT_Main/'
    train_dir = 'training_data/'
    data_directory = 'shapefile_data/'
    image_directory = 'image_data/'
    raster_dir = 'master_rasters'
    kernel_size = 57

    # for f in glob.glob(shp_dir + "*.shp"):
    #     fname = os.path.basename(f)
    #     split_shapefile(shp_dir, fname, data_directory)

    from pprint import pprint
    for irr in irrigated:
        for f in glob.glob(data_directory + "*.shp"):
            if "sample" not in f:
                if irr in f:
                    shp = os.path.basename(f)
                    others = get_all_shapefiles(shp, irrigated[irr], data_directory, irrigated)
                    pprint(others)
                    year = irrigated[irr]
                    class_code = 1
                    instances = 30000 
                    shp = os.path.basename(f)
                    n = shp[:-4] + "_train.h5"
                    if not os.path.isfile(os.path.join(train_dir, n)):
                        print("Creating training data for {}".format(n))
                        create_training_data(shp, data_directory, image_directory,
                                class_code=class_code, kernel_size=kernel_size, instances=instances, training_directory=train_dir, year=year, raster_directory=raster_dir) 
                    else:
                        print("Training data already created for", n)

                    for ff in others:
                        shp = os.path.basename(ff)
                        year = 2017
                        class_code = 0
                        n = shp[:-4] + "_train.h5"
                        if not os.path.isfile(os.path.join(train_dir, n)):
                            print("Creating training data for {}".format(n))
                            create_training_data(shp, data_directory, image_directory,
                                    class_code=class_code, kernel_size=kernel_size, instances=10000, training_directory=train_dir, year=year, raster_directory=raster_dir) 
                        else:
                            print("Training data already created for", n)
                   












