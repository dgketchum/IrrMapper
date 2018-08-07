# ===============================================================================
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
# ===============================================================================

import os
from pixel_classification.runspec import Montana, Nevada, Oregon, Utah, Washington
from pixel_classification.prepare_landsat import prepare_image_stack
from pixel_classification.compose_array import PixelTrainingArray
from pixel_classification.tf_multilayer_perceptron import mlp
from pixel_classification.classify import classify_raster

from numpy import vstack

home = os.path.expanduser('~')
ROOT = os.path.join(home, 'IrrigationGIS', 'western_states_irrgis')

OBJECT_MAP = {
    'MT': Montana,
    'NV': Nevada,
    'OR': Oregon,
    'UT': Utah,
    'WA': Washington
}


def build_training_feature_array():
    for key, obj in OBJECT_MAP.items():
        print(key)
        path = os.path.join(ROOT, key)
        geo = obj(path)
        if geo.sat == 8:
            prepare_image_stack(geo.path, geo.row, geo.year, path, geo.sat)
            p = PixelTrainingArray(path, instances=1000, overwrite_existing=True, geography=geo)
            p.extract_sample(save_points=True, limit_sample=False)


def build_model(model_location):
    first = True

    # concatenate data from all pickled PixelTrainingArray objects
    for key, obj in OBJECT_MAP.items():
        pkl_data = PixelTrainingArray(pickle_path=os.path.join(ROOT, key, 'data.pkl'))
        if first:
            training_data = {'data': pkl_data.data, 'target_values': pkl_data.target_values,
                             'features': pkl_data.features, 'model_map': pkl_data.model_map,
                             }
            first = False
        else:
            training_data = concatenate_training_data(training_data, pkl_data)

    p = PixelTrainingArray(from_dict=training_data)

    model_path = mlp(p, model_output=model_location)

    classify_raster()


def concatenate_training_data(existing, new_data):
    existing_array = existing['data']
    add_array = new_data.data
    new_array = vstack((existing_array, add_array))

    existing_labels = existing['target_values']
    add_labels = new_data.target_values
    new_lables = vstack((existing_labels.reshape((existing_labels.shape[0], 1)),
                         add_labels.reshape((add_labels.shape[0], 1))))

    concatenated = {'data': new_array, 'target_values': new_lables}
    return concatenated


if __name__ == '__main__':
    home = os.path.expanduser('~')
    model_loc = os.path.basename(__file__)
    build_model(model_loc)
    pass
# ========================= EOF ====================================================================
