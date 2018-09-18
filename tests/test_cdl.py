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
import unittest
import os
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio import open as rasopen

from sat_image.image import Landsat8
from bounds import RasterBounds

from pixel_classification.crop_data_layer import CropDataLayer as Cdl

TEST_DATA = os.path.join(os.path.dirname(__file__), 'data', 'cdl_data')


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.year = 2014
        self.target_profile = {'driver': 'GTiff',
                               'dtype': 'uint16',
                               'nodata': None,
                               'width': 7671,
                               'height': 7791,
                               'count': 1,
                               'crs': CRS({'init': 'epsg:32612'}),
                               'transform': Affine(30.0, 0.0, 582285.0,
                                                   0.0, -30.0, 5216715.0)}
        self.dir_name_LC8 = os.path.join(TEST_DATA, 'LC80370282013169LGN03')

    def test_download(self):
        cdl = Cdl(target_profile=self.target_profile, year=self.year)
        cdl.download_zipped_cdl()

    def test_conform_cdl(self):
        cdl = Cdl(target_profile=self.target_profile, year=self.year, out_dir=TEST_DATA)
        cdl.get_original_tif()

    def test_conforming_array(self):
        """ Test shape of Gridmet vs. Landsat image.
        :return:
        """
        l8 = Landsat8(self.dir_name_LC8)
        shape = 1, l8.rasterio_geometry['height'], l8.rasterio_geometry['width']
        polygon = l8.get_tile_geometry()
        cdl = Cdl(year=self.year, target_profile=l8.profile, out_dir=self.dir_name_LC8)
        _ = cdl.get_conforming_data(polygon)
        with rasopen(os.path.join(self.dir_name_LC8, 'cdl.tif')) as dst:
            arr = dst.read()

        self.assertEqual(arr.shape, shape)

    def test_mask(self):
        l8 = Landsat8(self.dir_name_LC8)
        shape = 1, l8.rasterio_geometry['height'], l8.rasterio_geometry['width']
        polygon = l8.get_tile_geometry()
        cdl = Cdl(from_file=os.path.join(self.dir_name_LC8, 'cdl.tif'))
        arr = cdl.get_mask(polygon, out_file=os.path.join(self.dir_name_LC8, 'cdl_mask.tif'))
        self.assertEqual(shape, arr.shape)
        self.assertEqual(arr.max(), 1)
        self.assertEqual(arr.min(), 0)


if __name__ == '__main__':
    unittest.main()
# ========================= EOF ====================================================================
