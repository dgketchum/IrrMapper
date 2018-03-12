# ===============================================================================
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
# ===============================================================================

import unittest

from fiona import open as fopen
from rasterio import open as rasopen

from pixel_classification.compose_array import point_raster_extract


class TestPointExtract(unittest.TestCase):
    def setUp(self):
        self.shapefile = 'tests/data/extract_test_attributed_Z12.shp'
        self.raster = 'tests/data/LE07_L1TP_039027_20130726_20160907_01_T1_B3_clip.tif'

    def tearDown(self):
        pass

    def test_raster_extract_by_point(self):
        """ Test native pet rasters vs. xarray netcdf point extract.
        :return: 
        """

        points = raster_point_extract(self.raster, self.shapefile)

        for key, val in points.items():
            self.assertEqual(val['raster_val'], val['extract_value'])


# ----------------------------------ANCILLARY FUNCTIONS-----------------------

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
    unittest.main()

# ===============================================================================
