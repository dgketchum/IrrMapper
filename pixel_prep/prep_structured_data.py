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

import os
import copy
import numpy as np
from pandas import get_dummies
from sklearn import decomposition
from sklearn.preprocessing import minmax_scale


class StructuredData(object):
    """ Structured data object for ML training, based on sklearn.utils.Bunch object"""

    def __init__(self, data):
        """

        :param data: dict object like {'features': }

        """
        # TODO add functionality that allows the creation of binary data using more than one True class

        self.lamda = None
        self.v = None

        self.data = data
        self.x = minmax_scale(self.data['data'].astype(np.float32),
                              feature_range=(-1, 1), axis=1)
        self.y_strs = self.data['target_values']

        unique, self.y = np.unique(self.y_strs, return_inverse=True)

        self.classes = unique
        self.class_counts = {x: list(self.y_strs).count(x) for x in self.classes}
        print('Class counts: {}'.format(self.class_counts))
        self.class_map = dict(zip(list(unique), list(range(len(unique)))))
        print('Class integer map: {}'.format(self.class_map))

        self.one_hot = get_dummies(self.y).values

    def make_binary(self, binary_true, inplace=False):
        """ Use a key value that will equate to True (1), all others to LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF."""
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
            self.x = pca.transform(self.x)
        elif return_percentile:
            pca = decomposition.PCA(return_percentile, copy=True, whiten=False)
            pca.fit(self.x)
            self.x = pca.transform(self.x)
            print('Cumulative sum principal components: {}\n '
                  '{} features \n {}'"%"' explained variance'.format(np.cumsum(pca.explained_variance_ratio_),
                                                                     pca.n_components_,
                                                                     pca.n_components * 100))
        return pca


if __name__ == '__main__':
    home = os.path.expanduser('~')


# ========================= EOF ================================================================
