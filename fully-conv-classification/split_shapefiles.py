import os
import argparse
from multiprocessing import Pool
from glob import glob
from shapefile_utils import filter_shapefile_overlapping


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shapefile-dir', 
            help='directory where unsplit shapefiles are stored', type=str)
    parser.add_argument('-o',  '--output-dir', help='where to store the split shapefiles', type=str)
    parser.add_argument('-p',  '--processes', help='n processes to spawn', default=12) 
    parser.add_argument('-e',  '--extension', help='shapefile extension', default=".shp") 
    args = parser.parse_args()
    fs = glob(os.path.join(args.shapefile_dir, "*" + args.extension))
    tf = [args.output_dir] * len(fs)
    with Pool() as pool:
        pool.starmap(filter_shapefile_overlapping, zip(fs, tf))
