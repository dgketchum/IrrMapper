# =============================================================================================
# Copyright 2017 dgketchum
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

import os
import numpy as np
from pandas import get_dummies


class StructuredData(object):
    def __init__(self, data, binary_true=False):
        self.data = data

        self.x = self.data['data'].astype(np.float32)
        y_strings = self.data['target_values']

        unique, self.y = np.unique(y_strings, return_inverse=True)
        self.classes = unique
        self.class_counts = {x: list(y_strings).count(x) for x in self.classes}
        print('Class counts: {}'.format(self.class_counts))
        self.class_map = dict(zip(list(unique), list(range(len(unique)))))
        print('Class integer map: {}'.format(self.class_map))

        if binary_true:
            self.y[y_strings == binary_true] = 1
            self.y[y_strings != binary_true] = 0

        self.one_hot = get_dummies(self.y).values


if __name__ == '__main__':
    home = os.path.expanduser('~')


# ========================= EOF ================================================================
