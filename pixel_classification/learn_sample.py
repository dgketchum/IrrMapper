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

from pixel_classification.tf_multilayer_perceptron import mlp
from pixel_classification.tf_softmax import softmax
from pixel_classification.compose_array import PixelTrainingArray


def classify(alg='mlp', data=None, path_to_pickled=None):
    if data:
        if not isinstance(data, PixelTrainingArray):
            raise TypeError('Classification requires a PixelTrainingArray object.')

    elif path_to_pickled:
        data = PixelTrainingArray(pickle_path=path_to_pickled)

    else:
        raise TypeError('Classification requires PixelTrainingArray object.')

    mapping = {'softmax': softmax,
               'mlp': mlp}

    try:
        cls = mapping[alg]
        results = cls(data)
        return None

    except KeyError:
        print('Invalid satellite key: "{}". available keys = {}'.format
              (alg, ', '.join(mapping.keys())))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    p_path = os.path.dirname(__file__).replace('pixel_classification', os.path.join('landsat_data', '39',
                                                                                    '27', '2015', 'data.pkl'))
    classify(alg='mlp', path_to_pickled=p_path)


# ========================= EOF ====================================================================
