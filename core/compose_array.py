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


def compose_data_array_():
    """ Compose numpy.ndarray prepped for a learning algorithm.
    
    
    Keyword Arguments:
    :param shapefile: .shp file from which point locations will be taken.
    :param raster: Single raster file path, list of files, or list of dirs, from which all
    /*.tif files will be used.
    :param transform: i.e., 'normalize', 'scale' data of real-number (continuous) variable
    :return: numpy.ndarray
    """
    pass


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

# ========================= EOF ====================================================================
