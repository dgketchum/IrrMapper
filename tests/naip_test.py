# =============================================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF (the "License");
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

import unittest

from rasterio import open as rasopen

from naip_image.naip import ApfoNaip


class AfpoNaipTestCase(unittest.TestCase):
    def setUp(self):
        self.box = (-109.9849, 46.46738, -109.93647, 46.498625)
        self.dst_srs = '26912'
        self.kwargs = dict([('dst_crs', self.dst_srs)])
        with rasopen('data/naip_test/wheatland_tile.tif', 'r') as src:
            self.profile = src.profile
            self.array = src.read()

    def test_image_reference(self):
        naip = ApfoNaip(self.box, **self.kwargs)
        array, profile = naip.get_image('montana')
        self.assertAlmostEqual(array.mean(), self.array.mean(), delta=3.)
        naip.close()


if __name__ == '__main__':
    unittest.main()

# ========================= EOF ====================================================================
