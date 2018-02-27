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


def raster_point_extract(raster, points, dtime):
    """ Get point values from a raster.
    
    This function opens a points shapefile, gets the coordinates of the points, 
    opens a raster as an np.array object, and finds the row, col of the cell covering
    that point.
    
    :param raster: local_raster: Thredds.Gridmet-derived array in Landsat image geometry.
    :param points: Shapefile of points.
    :param dtime: Datetime.datetime object.
    :return: Dict of coords, row/cols, and values of raster at that point.
    """
    point_data = {}
    with fopen(points, 'r') as src:
        for feature in src:
            name = feature['properties']['siteid']
            point_data[name] = {'coords': feature['geometry']['coordinates']}

        with rasopen(raster, 'r') as rsrc:
            rass_arr = rsrc.read()
            rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
            affine = rsrc.transform

        for key, val in point_data.items():
            x, y = val['coords']
            col, row = ~affine * (x, y)
            val = rass_arr[int(row), int(col)]
            point_data[key][dtime] = [val, None]

        return point_data


if __name__ == '__main__':
    home = os.path.expanduser('~')

# ========================= EOF ====================================================================
