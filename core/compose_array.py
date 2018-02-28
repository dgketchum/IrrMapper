# =============================================================================================
# Copyright 2017 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================================

import os

from fiona import open as fopen
from rasterio import open as rasopen

'''
This script contains functions meant to gather data from rasters using a points shapefile.  The high-level 
function `compose_data_array` will return a numpy.ndarray object ready for a learning algorithm.  
'''


def load_irrigation_data(shapefile, rasters, target_field='LType', bands=[]):
    """ Compose numpy.ndarray prepped for a learning algorithm.
    
    
    Keyword Arguments:
    :param shapefile: .shp file from which point locations will be taken.
    :param raster: Single raster file path, list of file paths, or a dir, from which all
    /*.tif files will be used.
    :param transform: i.e., 'normalize', 'scale' data of real-number (continuous) variable
    :return: numpy.ndarray
    """
    target_names = None
    target = None
    data = None
    raster_list = raster_paths(rasters)
    for r in raster_list:
        point_data = raster_point_extract(r, shapefile)

    return data, target, target_names


def raster_paths(rasters):
    """ Return list of rasters from single raster, list of rasters, or a dir.    """
    if os.path.isfile(rasters):
        return [rasters]
    elif os.path.isfile(rasters[0]):
        return rasters
    elif os.path.isdir(rasters):
        return list(recursive_file_gen(rasters))

    else:
        raise ValueError('Must provide a single .tif, a list of .tif files, or a dir')


def recursive_file_gen(mydir):
    for root, dirs, files in os.walk(mydir):
        for file in files:
            if file.endswith('.tif'):
                yield os.path.join(root, file)


def raster_point_extract(raster, points):
    """ Get point values from a raster.
    
    :param raster: local_raster
    :param points: Shapefile of points.
    :return: Dict of coords, row/cols, and values of raster at that point.
    """
    point_data = {}

    with fopen(points, 'r') as src:
        for feature in src:
            name = feature['id']
            proj_coords = feature['geometry']['coordinates']

            point_data[name] = {'coords': proj_coords,
                                'label': feature['properties']['LType'],
                                'raster_val': int(feature['properties']['LE07_L1TP_'])}

    with rasopen(raster, 'r') as rsrc:
        rass_arr = rsrc.read()
        rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
        affine = rsrc.affine

    for key, val in point_data.items():
        x, y = val['coords']
        col, row = ~affine * (x, y)
        raster_val = rass_arr[int(row), int(col)]
        val['extract_value'] = raster_val

    return point_data


if __name__ == '__main__':
    home = os.path.expanduser('~')
    montana = os.path.join(home, 'images', 'irrigation', 'MT')
    images = os.path.join(montana, 'landsat')
    shape = os.path.join(montana, 'hex_centroids_1000m_intersect.shp')
    load_irrigation_data(shape, images)

# ========================= EOF ====================================================================
