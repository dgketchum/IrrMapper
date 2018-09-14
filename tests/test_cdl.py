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

from rasterio.crs import CRS
from rasterio.transform import Affine

from pixel_classification.crop_data_layer import CropDataLayer as Cdl

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

    def test_download(self):
        cdl = Cdl(target_profile=self.target_profile, year=self.year)
        cdl.download_zipped_cdl()


if __name__ == '__main__':
    unittest.main()
# ========================= EOF ====================================================================
