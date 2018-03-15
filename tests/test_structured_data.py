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
import pickle
from numpy import array, any
from pixel_prep.prep_structured_data import StructuredData


class StructuredDataTest(unittest.TestCase):
    def setUp(self):
        path_to_pickled = 'data/test.pkl'
        with open(path_to_pickled, 'rb') as p:
            data = pickle.load(p)

        self.struct = StructuredData(data)

    def test_data_instant(self):
        # this assertion is really a test of the test which in general is not done
        # self.assertIsInstance(self.struct, StructuredData)
        self.assertEquals(self.struct.class_counts['I'], 4)

    def test_class_zero(self):
        classes = self.struct.classes
        self.assertEquals(classes[0], 'F')
        # i try to only do one assert per test
        # self.assertAlmostEqual(pca.mean_[0], 50.)

    def test_data_pca_mean(self):
        pca = self.struct.principal_components(n_components=1)
        self.assertAlmostEqual(pca.mean_[0], 50.)

    def test_data_binary(self):
        self.struct.make_binary('I', inplace=True)

        # use self.assert...?
        # assert (self.struct.one_hot == array([[0, 1],
        #                                       [0, 1],
        #                                       [0, 1],
        #                                       [1, 0],
        #                                       [0, 1], ])).any()
        expected = array([[0, 1],
                          [0, 1],
                          [0, 1],
                          [1, 0],
                          [0, 1]])

        self.assertListEqual(list(self.struct.one_hot), expected)


if __name__ == '__main__':
    unittest.main()

# ========================= EOF ====================================================================
