# ===============================================================================
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
# ===============================================================================

import os
import shutil
import unittest

from fiona import open as fopen

from pixel_classification.compose_array import PixelTrainingArray
from tests.build_extract_test_data import make_test_dataset
from pixel_classification.training_keys import return_object


class TestPointExtract(unittest.TestCase):
    def setUp(self):
        self.origin = os.path.join(os.path.dirname(__file__), 'data', 'pixel_extract_test')
        self.satellite = 'LC8'
        self.directory = os.path.join(os.path.dirname(__file__), 'data', 'pixel_extract_test')

    # def tearDown(self):
    #     shutil.rmtree(self.directory)

    def test_sample_points(self):

        montana = return_object('montana')
        p = PixelTrainingArray(images=self.directory, instances=100, geography=montana)
        p.extract_sample(save_points=True)
        with fopen(p.shapefile_path, 'r') as src:
            points = [x for x in src]
        self.assertGreater(len(points), 100)
        self.assertGreater(p.extracted_points.shape[0], 100)
        self.assertEqual(p.extracted_points.shape[1], 8)

    def test_instantiate_w_pkl(self):
        p = PixelTrainingArray(pickle_path=self.pkl)
        self.assertIsInstance(p, PixelTrainingArray)


if __name__ == '__main__':
    unittest.main()

# ===============================================================================
