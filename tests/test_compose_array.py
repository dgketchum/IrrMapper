# ===============================================================================
# Copyright 2018 dgketchum
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

import os
import unittest

from fiona import open as fopen
from rasterio import open as rasopen

from pixel_prep.compose_array import point_target_extract


class TestPointExtract(unittest.TestCase):
    def setUp(self):
        self.shapefile = 'data/extract_test_attributed_Z12.shp'
        self.raster = 'data/LE07_L1TP_039027_20130726_20160907_01_T1_B3_clip.tif'
        if not os.path.isfile(self.shapefile):
            raise ValueError('Path to shapefile is invalid')

    def tearDown(self):
        pass

    def test_raster_extract_by_point(self):
        """ Test native pet rasters vs. xarray netcdf point extract.
        :return: 
        """

        points = point_target_extract(self.shapefile, nlcd_path=None,
                                      )

        for key, val in points.items():
            self.assertEqual(val['raster_val'], val['extract_value'])



if __name__ == '__main__':
    unittest.main()

# ===============================================================================
