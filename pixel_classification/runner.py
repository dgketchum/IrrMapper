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

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)
from numpy import vstack

from pixel_classification.runspec import Montana, Nevada, Oregon, Utah, Washington
from pixel_classification.prepare_images import ImageStack
from pixel_classification.compose_array import PixelTrainingArray as Pta
from pixel_classification.tf_multilayer_perceptron import mlp
from pixel_classification.classify import classify_multiproc

OBJECT_MAP = {'MT': Montana,
              'NV': Nevada,
              'OR': Oregon,
              'UT': Utah,
              'WA': Washington}


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

    existing['paths_map'].update(new_data.paths_map)

    concatenated = {'data': new_array, 'target_values': new_lables, 'features': new_features,
                    'paths_map': existing['paths_map']}

    return concatenated


def model_training_scenes(project, n_images, training):
    training_data = {}
    for key, val in OBJECT_MAP.items():

        first = True

        print('Train on {}'.format(key))

        project_state_dir = os.path.join(project, key)

        if not os.path.isdir(project_state_dir):
            os.mkdir(project_state_dir)

        geography = os.path.join(training, key)
        geo = val(geography)
        geo_folder = os.path.join(project, key)
        geo_data_path = os.path.join(geo_folder, 'data.pkl')

        i = ImageStack(root=project_state_dir, satellite=geo.sat, path=geo.path, row=geo.row,
                       n_landsat=n_images, year=geo.year, max_cloud_pct=70)
        i.build_all()
        p = Pta(root=i.root, geography=geo, paths_map=i.model_map, instances=100,
                overwrite_array=False, overwrite_points=False, pkl_path=geo_data_path)
        p.extract_sample()

        geo = val(project)
        pkl_data = Pta(root=project, geography=geo, pkl_path=os.path.join(root, 'data.pkl'))
        if first:
            training_data = {'data': pkl_data.data, 'target_values': pkl_data.target_values,
                             'features': pkl_data.features, 'model_map': pkl_data.paths_map,
                             }
            first = False
        else:
            training_data = concatenate_training_data(training_data, pkl_data)

        print('Shape {}: {}'.format(key, pkl_data.data.shape))

    p = Pta(from_dict=None)
    p.to_pickle(training_data, path)
    model_name = os.path.join(project_dir, 'model.ckpt')
    mlp(p, model_name)
    print('Model saved to {}'.format(model_name))


def classify_scene(path, row, sat, year, eval_directory, model, result=None):
    sub = os.path.join(eval_directory, '{}_{}_{}'.format(path, row, year))
    if not os.path.isdir(sub):
        os.mkdir(sub)

    i = ImageStack(root=sub, satellite=sat, path=path, row=row,
                   n_landsat=3, year=year, max_cloud_pct=70)
    i.build_all()

    if not result:
        result = '{}{}{}_{}.tif'.format(i.sat_abv, path, row, year)

    classify_multiproc(model, stack_data=i, mask=i.cdl_mask, result=result)


if __name__ == '__main__':
    home = os.path.expanduser('~')

    training_dir = os.path.join(home, 'IrrigationGIS', 'western_states_irrgis')
    classified_dir = os.path.join(home, 'IrrigationGIS', 'classified')
    model_data = os.path.join(abspath, 'model_data')
    project_dir = os.path.join(model_data, 'allstates_1')

    model_training_scenes(project_dir, 1, training_dir)
    # classify_scene(path=39, row=27, sat=8, year=2015, eval_directory=stacks, model=model_name)
# ========================= EOF ====================================================================
