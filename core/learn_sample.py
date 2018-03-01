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
import pickle

from core.softmax import softmax_regression
from core.neural_network import neural_net


def classify(alg='softmax', data=None, path_to_pickled=None):

    if data:
        pass
    elif path_to_pickled:
        with open(path_to_pickled, 'rb') as p:
            data = pickle.load(p)
    else:
        dct_form = "'{'classes': target.unique, 'data': numpy.ndarray,\n " \
                   "'target_values': target}'"
        raise ValueError('\nMust provide data for classification.  '
                         'This can \neither beas a dict or '
                         'pickled dict of form:\n{}'.format(dct_form))

    mapping = {'softmax': softmax_regression(data, binary_true='I'),
               'neural_net': neural_net()}

    try:
        cls = mapping[alg]
        results = cls(data)

    except KeyError:
        print('Invalid satellite key: "{}". available keys = {}'.format
              (alg, ','.join(mapping.keys())))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    spatial = os.path.join(home, 'PycharmProjects', 'IrrMapper', 'spatial')
    p_path = os.path.join(spatial, 'pick.pickle')
    classify(alg='softmax', path_to_pickled=p_path)


# ========================= EOF ====================================================================
