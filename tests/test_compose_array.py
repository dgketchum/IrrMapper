# ===============================================================================
# Copyright 2018 dgketchum
#
<<<<<<< HEAD
# Licensed under the Apache License, Version 2.LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF (the "License");
=======
# Licensed under the Apache License, Version 2.(the "License");
>>>>>>> ce638691ecdf01bbf7b8706544f7c020cf211067
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
<<<<<<< HEAD
import unittest

from pixel_prep.compose_array import make_data_array
=======
import shutil
import unittest

from fiona import open as fopen

from pixel_classification.compose_array import PixelTrainingArray
from tests.build_extract_test_data import make_test_dataset
from pixel_classification.training_keys import return_object
>>>>>>> ce638691ecdf01bbf7b8706544f7c020cf211067


class TestPointExtract(unittest.TestCase):
    def setUp(self):
<<<<<<< HEAD
        self.shapefile = 'data/extract_no_attrs_z12.shp'
        self.raster = 'data/LE07_clip_32612_L1TP_039027_20130726_20160907_01_T1_B3.TIF'
        self.nlcd = 'data/nlcd_clip_test.tif'
        self.target_polys = 'data/flu_test_z12.shp'
        if not os.path.isfile(self.shapefile):
            raise ValueError('Path to shapefile is invalid')

    def tearDown(self):
        pass

    def test_compose_array(self):
        """ Test native pet rasters vs. xarray netcdf point extract.
        :return: 
        """

        points = make_data_array(self.shapefile, self.raster, nlcd_path=self.nlcd, target_shapefiles=self.target_polys)

        self.assertEqual(points['target_values'][0], ['I', 'I', 'I', 'F', 'I'][0])
        self.assertEqual(points['data'][0], [63, 51, 54, 82, 0][0])
        self.assertEqual(points['features'][0], '039027_T1')
=======
        self.origin = os.path.join(os.path.dirname(__file__), 'data', 'pixel_extract_test')
        self.satellite = 'LC8'
        self.directory = os.path.join(os.path.dirname(__file__), 'data', 'pixel_extract_test')

    def test_sample_points(self):
        montana = return_object('montana_test')
        p = PixelTrainingArray(images=self.directory, instances=10,
                               geography=montana, overwrite_existing=True)
        p.extract_sample(save_points=True, limit_sample=True)
        with fopen(p.shapefile_path, 'r') as src:
            points = [x for x in src]
        self.assertGreater(len(points), 40)
        self.assertGreater(p.extracted_points.shape[0], 40)

    def test_instantiate_w_pkl(self):
        p = PixelTrainingArray(pickle_path=self.pkl)
        self.assertIsInstance(p, PixelTrainingArray)
>>>>>>> ce638691ecdf01bbf7b8706544f7c020cf211067


if __name__ == '__main__':
    unittest.main()

# ===============================================================================
