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

from pixel_classification.compose_array import PixelTrainingArray
from pixel_classification.tf_multilayer_perceptron import mlp


def build_model(data, model=None):
    if isinstance(data, PixelTrainingArray):
        pass

    elif os.path.isfile(data):
        data = PixelTrainingArray(pickle_path=data)

    else:
        raise TypeError('Classification requires PixelTrainingArray object.')

    mlp(data, model)

    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.dirname(__file__).replace('pixel_classification', 'landsat_data')

    year = os.path.join(root, '39', '27', '2015')
    p_path = os.path.join(year, 'data.pkl')

    build_model(p_path, model=year)

# ========================= EOF ====================================================================
