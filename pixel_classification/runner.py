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
from datetime import datetime

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


def build_model(project_root, path, model_path):
    first = True
    for key, obj in OBJECT_MAP.items():
        root = os.path.join(project_root, key)
        geo = obj(root)
        pkl_data = Pta(root=root, geography=geo, pkl_path=os.path.join(root, 'data.pkl'))
        if first:
            training_data = {'data': pkl_data.data, 'target_values': pkl_data.target_values,
                             'features': pkl_data.features, 'model_map': pkl_data.model_map,
                             }
            first = False
        else:
            training_data = concatenate_training_data(training_data, pkl_data)

        print('Shape {}: {}'.format(key, pkl_data.data.shape))

    p = Pta(from_dict=training_data)
    p.to_pickle(training_data, path)
    model_path = mlp(p, model_path)

    return model_path


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


def run_training_scenes(model, project, training=None):

    for key, val in OBJECT_MAP.items():
        print('Classify {}'.format(key))

        project_state_dir = os.path.join(project, key)

        if not os.path.isdir(project_state_dir):
            os.mkdir(project_state_dir)

        geography = os.path.join(training, key)
        geo = val(geography)

        if training:
            i = ImageStack(root=project_state_dir, satellite=geo.sat, path=geo.path, row=geo.row,
                           n_landsat=3, year=geo.year, max_cloud_pct=70).build_all()
            p = Pta(root=i.root, geography=geo, instances=100,
                    overwrite_array=False, overwrite_points=False).extract_sample()
            i.warp_vrt()

        if not model:
            build_model(project, data_path, model)

        tif_name = '{}_{}{}.tif'.format(key.lower(), datetime.now().month, datetime.now().day)
        classified_tif = os.path.join(classified_dir, tif_name)
        geo_folder = os.path.join(project, key)
        save_array = os.path.join(geo_folder, 'array.npy')
        geo_data_path = os.path.join(geo_folder, 'data.pkl')
        cdl_path = os.path.join(geo_folder, 'cdl_mask.tif')

        pta = Pta()
        pta.from_pickle(path=geo_data_path)
        classify_multiproc(model, pta, array_outfile=save_array,
                           mask=cdl_path, result=classified_tif)
    return None


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

    # stacks = os.path.join(model_data, 'stacks')

    project_dir = os.path.join(model_data, 'allstates_1')
    data_path = os.path.join(project_dir, 'data.pkl')
    model_name = os.path.join(project_dir, 'model.ckpt')

    run_training_scenes(model_name, project_dir, training=training_dir)
    # classify_scene(path=39, row=27, sat=8, year=2015, eval_directory=stacks, model=model_name)
# ========================= EOF ====================================================================
