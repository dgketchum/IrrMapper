# =============================================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.(the "License");
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
import unittest
from rasterio import open as rasopen

from naip_image.naip import ApfoNaip


class AfpoNaipTestCase(unittest.TestCase):
    def setUp(self):
        self.box = (-109.9849, 46.46738, -109.93647, 46.498625)
        self.dst_srs = '26912'
        self.kwargs = dict([('dst_crs', self.dst_srs)])
        self.kwargs_centroid = dict([('dst_crs', self.dst_srs), ('centroid', ((self.box[1] + self.box[3]) / 2,
                                                                 ((self.box[0] + self.box[2]) / 2))),
                                     ('buffer', 1700)])
        self.tile_loc = os.path.join(os.path.dirname(__file__), 'data', 'wheatland_tile.tif')

        with rasopen(self.tile_loc, 'r') as src:
            self.profile = src.profile
            self.array = src.read()

    def test_image_reference_bbox(self):
        naip = ApfoNaip(self.box, **self.kwargs)
        array, profile = naip.get_image('MT')
        self.assertAlmostEqual(array.mean(), self.array.mean(), delta=3.)
        naip.close()

    def test_image_reference_centroid(self):
        naip = ApfoNaip(**self.kwargs_centroid)
        array, profile = naip.get_image('MT')
        naip.save(array, profile, self.tile_loc.replace('wheatland_tile', 'centroid_test'))
        self.assertAlmostEqual(array.mean(), self.array.mean(), delta=3.)
        naip.close()


if __name__ == '__main__':
    unittest.main()

# ========================= EOF ====================================================================
