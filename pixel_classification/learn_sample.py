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
import pickle

from pixel_classification.prep_structured_data import StructuredData
from pixel_classification.tf_multilayer_perceptron import mlp
from pixel_classification.tf_softmax import softmax


def classify(alg='mlp', data=None, path_to_pickled=None,
             binary=None):
    if data:
        pass
    elif path_to_pickled:
        with open(path_to_pickled, 'rb') as p:
            data = pickle.load(p)
    else:
        dct_form = "'{'classes': target.unique, 'data': numpy.ndarray,\n " \
                   "'target_values': target}'"
        raise ValueError('\nMust provide data for classification.  '
                         'This can \neither be a dict or '
                         'pickled dict of form:\n{}'.format(dct_form))

    data = StructuredData(data)
    data.make_binary(binary_true=binary, inplace=True)
    data.principal_components(return_percentile=0.90)

    mapping = {'softmax': softmax,
               'mlp': mlp}

    try:
        cls = mapping[alg]
        results = cls(data)

    except KeyError:
        print('Invalid satellite key: "{}". available keys = {}'.format
              (alg, ', '.join(mapping.keys())))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    spatial = os.path.join(home, 'PycharmProjects', 'IrrMapper', 'spatial_data')
    p_path = os.path.join(spatial, 'P39R27_Test_all.pkl')
    classify(alg='neural_net', path_to_pickled=p_path, binary='I')


# ========================= EOF ====================================================================
