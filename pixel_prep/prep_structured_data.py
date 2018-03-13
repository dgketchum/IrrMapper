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

import os
import copy
import numpy as np
from pandas import get_dummies
from sklearn import decomposition


class StructuredData(object):
    """ Structured data object for ML training, not unlike a sklearn.dataset.load_dataset object"""

    def __init__(self, data):
        self.lamda = None
        self.v = None

        self.data = data

        self.x = self.data['data'].astype(np.float32)
        self.y_strs = self.data['target_values']

        unique, self.y = np.unique(self.y_strs, return_inverse=True)

        self.classes = unique
        self.class_counts = {x: list(self.y_strs).count(x) for x in self.classes}
        print('Class counts: {}'.format(self.class_counts))
        self.class_map = dict(zip(list(unique), list(range(len(unique)))))
        print('Class integer map: {}'.format(self.class_map))

        self.one_hot = get_dummies(self.y).values

    def make_binary(self, binary_true, inplace=False):
        """ Use a key value that will equate to True (1), all others to 0."""
        """
        :param binary_true: 
        :return: 
        """
        if inplace:
            self.y[self.y_strs == binary_true] = 1
            self.y[self.y_strs != binary_true] = 0
            self.y_strs[self.y_strs != binary_true] = '{}{}'.format('N', binary_true)
            unique, _ = np.unique(self.y_strs, return_inverse=True)
            self.classes = unique
            self.class_counts = {x: list(self.y_strs).count(x) for x in self.classes}
            self.one_hot = get_dummies(self.y).values
        else:
            new = copy.deepcopy(self)
            self.make_binary(binary_true, inplace=True)
            return new

    def principal_components(self, return_percentile=None, n_components=None):
        """ Extract eigenvectors and eigenvalue, return desired PCAs""
        :return:
        """
        if n_components:
            pca = decomposition.PCA(n_components=n_components, copy=True, whiten=False)
            pca.fit(self.x)
        elif return_percentile:
            pca = decomposition.PCA(0.95, copy=True, whiten=False)
            pca.fit(self.x)

        print (np.cumsum(pca.explained_variance_ratio_))
        return pca


if __name__ == '__main__':
    home = os.path.expanduser('~')


# ========================= EOF ================================================================
