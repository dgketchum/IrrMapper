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

from rasterio import open as rasopen
from numpy import zeros, uint16, linspace

from pixel_classification.tf_multilayer_perceptron import mlp
from pixel_classification.tf_softmax import softmax
from pixel_classification.compose_array import PixelTrainingArray


def apply_model(model, pixel_data):
    data = PixelTrainingArray(pickle_path=pixel_data)

    lows = linspace(0, 7000, 8).tolist()
    highs = linspace(1000, 8000, 8).tolist()

    stack = []

    for feat in data.features.tolist():
        # print('Block {} of 64'.format(i + 1))
        with rasopen(data.model_map[feat], mode='r') as src:
            arr = src.read()
        empty = zeros((1, 8000, 8000), dtype=uint16)
        try:
            empty[0, :arr.shape[1], :arr.shape[2]] = arr
        except ValueError:
            pass
        padded = empty
        stack.append(padded)
    pass


def build_model(data, alg='mlp', model=None):
    if isinstance(data, PixelTrainingArray):
        pass

    elif os.path.isfile(data):
        data = PixelTrainingArray(pickle_path=data)

    else:
        raise TypeError('Classification requires PixelTrainingArray object.')

    mapping = {'softmax': softmax,
               'mlp': mlp}

    try:
        cls = mapping[alg]
        cls(data, model)

    except KeyError:
        print('Invalid algorithm key: "{}". available keys = {}'.format
              (alg, ', '.join(mapping.keys())))

    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')
    p_path = os.path.dirname(__file__).replace('pixel_classification', os.path.join('landsat_data', '39',
                                                                                    '27', '2015', 'data.pkl'))
    checkpoint = p_path.replace('data.pkl', 'checkpoint.chk')
    # build_model(p_path, alg='mlp', model=checkpoint)
    apply_model(None, p_path)

# ========================= EOF ====================================================================
