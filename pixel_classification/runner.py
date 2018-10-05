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
from pixel_classification.runspec import get_path_rows, get_selected_path_rows
from pixel_classification.prepare_images import ImageStack
from pixel_classification.compose_array import PixelTrainingArray as Pta
from pixel_classification.tf_multilayer_perceptron import mlp
from pixel_classification.classify import classify_multiproc

OBJECT_MAP = {'MT': Montana,
              'NV': Nevada,
              'OR': Oregon,
              'UT': Utah,
              'WA': Washington}


def concatenate_training_data(existing, training_array):
    existing_array = existing['data']
    add_array = training_array.data
    new_array = vstack((existing_array, add_array))

    existing_features = existing['features']
    add_features = training_array.features
    new_features = vstack((existing_features.reshape((existing_features.shape[0], 1)),
                           add_features.reshape((add_features.shape[0], 1))))

    existing_labels = existing['target_values']
    add_labels = training_array.target_values
    new_lables = vstack((existing_labels.reshape((existing_labels.shape[0], 1)),
                         add_labels.reshape((add_labels.shape[0], 1))))

    concatenated = {'data': new_array, 'target_values': new_lables, 'features': new_features}

    return concatenated


def model_training_scenes(project, n_images, training, model):
    training_data = {}
    first = True

    for key, val in OBJECT_MAP.items():

        print('Train on {}'.format(key))

        project_state_dir = os.path.join(project, key)

        if not os.path.isdir(project_state_dir):
            try:
                os.mkdir(project_state_dir)
            except FileNotFoundError:
                os.makedirs(project_state_dir)

        geography = os.path.join(training, key)
        geo = val(geography)
        geo_folder = os.path.join(project, key)
        geo_data_path = os.path.join(geo_folder, 'data.pkl')

        if not os.path.isfile(geo_data_path):
            geo_data_path = None

        i = ImageStack(root=project_state_dir, satellite=geo.sat, path=geo.path, row=geo.row,
                       n_landsat=n_images, year=geo.year, max_cloud_pct=70)
        i.build_training()
        p = Pta(root=i.root, geography=geo, paths_map=i.paths_map, instances=10000, masks=i.masks,
                overwrite_array=False, overwrite_points=False, pkl_path=geo_data_path)
        p.extract_sample()

        if first:
            training_data = {'data': p.data, 'target_values': p.target_values,
                             'features': p.features}
            first = False
        else:
            training_data = concatenate_training_data(training_data, p)

        print('Shape {}: {}'.format(key, p.data.shape))

    p = Pta(from_dict=training_data)
    p.to_pickle(training_data, os.path.join(project, 'data.pkl'))
    mlp(p, model)
    print('Model saved to {}'.format(model))


def classify_scene(path, row, sat, year, eval_directory, model, n_images, n_classes, result=None):
    print('Time: {}'.format(datetime.now()))
    print('Classfiy path {} row {} sat {} year {}'.format(path, row, sat, year))
    sub = os.path.join(eval_directory, '{}_{}_{}'.format(path, row, year))
    if not os.path.isdir(sub):
        os.mkdir(sub)
    # try:
    i = ImageStack(root=sub, satellite=sat, path=path, row=row,
                   n_landsat=n_images, year=year, max_cloud_pct=70)
    i.build_evaluating()
    i.warp_vrt()

    if not result:
        tif = '{}{}{}_{}_{}c_{}i.tif'.format(i.sat_abv, path, row, year, n_classes, n_images)
        path_row_year_dir = '{}_{}_{}'.format(path, row, year)
        result = os.path.join(eval_directory, path_row_year_dir, tif)

    # classify_multiproc(model, stack_data=i, mask=i.cdl_mask, result=result)
    print('Time: {}'.format(datetime.now()))
    # except Exception as e:
    #     print(e)
    #     print('')


def run_targets(directory, model, classes):
    prs = get_selected_path_rows()
    years = [2015]
    for (p, r) in prs:
        for yr in years:
            print('')
            print('Classify path {} row {} year {}'.format(p, r, yr))
            print('')
            classify_scene(p, r, 8, yr, directory, model, 3, n_classes=classes)


if __name__ == '__main__':
    home = os.path.expanduser('~')

    training_dir = os.path.join(home, 'IrrigationGIS', 'western_states_irrgis')
    model_data = os.path.join(abspath, 'model_data')
    model_name_2 = os.path.join(model_data, 'model-2c-3i.ckpt')
    model_name_4 = os.path.join(model_data, 'model-3.ckpt')
    t_project_dir = os.path.join(model_data, 'allstates_2c_3i')
    stack = os.path.join(home, 'data_mt')

    c_project_dir = os.path.join(stack, 'classified')

    # model_training_scenes(t_project_dir, n_images, training_dir, model_name)
    # classify_scene(path=39, row=27, sat=8, year=2015,
    #                eval_directory=c_project_dir, n_images=3, model=model_name)
    run_targets(stack, model_name_2, classes=2)
    run_targets(stack, model_name_4, classes=4)
# ========================= EOF ====================================================================
