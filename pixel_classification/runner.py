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
import sys
from runspec import Montana, Nevada, Oregon, Utah, Washington
from prepare_landsat import prepare_image_stack
from compose_array import PixelTrainingArray
from tf_multilayer_perceptron import mlp
from classify import classify_stack

from sat_image.image import LandsatImage

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
            # prepare_image_stack(geo.path, geo.row, geo.year, path, geo.sat)
            p = PixelTrainingArray(path, instances=1100, overwrite_existing=True, geography=geo)
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
    mlp(p, model_output=model_location)

    for key, obj in OBJECT_MAP.items():

        dst = os.path.join(ROOT, key, 'classified_rasters')
        if not os.path.isdir(dst):
            os.mkdir(dst)


def classify_rasters(model):
    for key, obj in OBJECT_MAP.items():
        path = os.path.join(ROOT, key)
        geo = obj(path)
        dst = os.path.join(geo.root, str(geo.path), str(geo.row), str(geo.year))
        meta = StackMetadata(dst)
        classify_stack(meta, model=model, out_location=dst)


class StackMetadata(object):

    def __init__(self, directory):
        self.dirs = [os.path.join(directory, x) for x in os.listdir(directory) if os.path.isdir(os.path.join(directory,
                                                                                                             x))]
        self.metadata = {}
        file_list = []

        for d in self.dirs:
            l = LandsatImage(d)
            self.metadata[d] = [os.path.join(d, x) for x in l.tif_list if 'BQA' not in x]
            file_list.append(self.metadata[d])

        self.file_list = [item for sublist in file_list for item in sublist]
        self.stack_shape = (len(self.file_list), l.rasterio_geometry['height'], l.rasterio_geometry['width'])


def concatenate_training_data(existing, new_data):
    existing_array = existing['data']
    add_array = new_data.data
    new_array = vstack((existing_array, add_array))

    existing_features = existing['features']
    add_features = new_data.features
    new_features = vstack((existing_features.reshape((existing_features.shape[0], 1)),
                           add_features.reshape((add_features.shape[0], 1))))

    existing_labels = existing['target_values']
    add_labels = new_data.target_values
    new_lables = vstack((existing_labels.reshape((existing_labels.shape[0], 1)),
                         add_labels.reshape((add_labels.shape[0], 1))))

    existing['model_map'].update(new_data.model_map)

    concatenated = {'data': new_array, 'target_values': new_lables, 'features': new_features,
                    'model_map': existing['model_map']}

    return concatenated


if __name__ == '__main__':
    home = os.path.expanduser('~')
    model_loc = os.path.join(os.path.dirname(__file__), 'classifier')
    build_training_feature_array()
    build_model(model_loc)
    classify_rasters(model_loc)
# ========================= EOF ====================================================================
