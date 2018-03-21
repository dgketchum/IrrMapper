# =============================================================================================
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
# =============================================================================================

import unittest

from convnet.naip import ApfoNaip


class AfpoNaipTestCase(unittest.TestCase):
    def setUp(self):
        self.tile_size = (512, 512)
        self.box = (-109.9849, 46.46738, -109.93647, 46.498625)
        self.test_tif = 'data/test_apfonaip.tif'
        self.dst_srs = 26912
        self.profile = dict([('dst_crs', 26912)])

    def test_image_reference(self):
        naip = ApfoNaip(self.box, **self.profile)
        array, profile = naip.get_image('montana')
        naip.save(array, profile, self.test_tif, crs=self.dst_srs)
        pass


if __name__ == '__main__':
    unittest.main()

# ========================= EOF ====================================================================
