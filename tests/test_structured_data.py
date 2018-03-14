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

from sklearn.datasets import load_iris

from pixel_prep.prep_structured_data import StructuredData


class StructuredDataTest(unittest.TestCase):
    def setUp(self):
        self.test_data = load_iris()
        self.input = {'features': self.test_data.target_names, 'data': self.test_data.data,
                      'target_values': self.test_data.target}
        self.struct = StructuredData(self.input)

    def test_data_structure(self):
        self.assertIsInstance(self.struct, StructuredData)
        self.assertEquals(self.struct.class_counts[0], 50)
        classes = self.struct.classes
        pca = self.struct.principal_components(return_percentile=0.95)
        self.assertEquals(classes[0], 0)
        self.assertAlmostEqual(pca.explained_variance_[0], 4.2248406)


if __name__ == '__main__':
    unittest.main()

# ========================= EOF ====================================================================
