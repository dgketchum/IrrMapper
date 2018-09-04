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
from numpy import vstack, array_split, concatenate
# from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import cpu_count
from multiprocess.pool import Pool

from pixel_classification.runspec import Montana, Nevada, Oregon, Utah, Washington
from pixel_classification.prepare_landsat import prepare_image_stack
from pixel_classification.compose_array import PixelTrainingArray
from pixel_classification.tf_multilayer_perceptron import mlp
from pixel_classification.classify import Classifier

home = os.path.expanduser('~')
ROOT = os.path.join(home, 'IrrigationGIS', 'western_states_irrgis')

OBJECT_MAP = {
    'MT': Montana,
    # 'NV': Nevada,
    # 'OR': Oregon,
    # 'UT': Utah,
    # 'WA': Washington
}


def build_training_feature_array(skip_landsat=False):
    for key, obj in OBJECT_MAP.items():
        print(key)
        path = os.path.join(ROOT, key)
        geo = obj(path)
        if geo.sat == 8:
            prepare_image_stack(geo.path, geo.row, geo.year, path, geo.sat,
                                skip_landsat=skip_landsat)
            dem = os.path.join(path, str(geo.path), str(geo.row), 'dem.tif')
            slope = os.path.join(path, str(geo.path), str(geo.row), 'slope.tif')
            p = PixelTrainingArray(root=path, geography=geo, instances=5000, overwrite_array=True,
                                   overwrite_points=False, ancillary_rasters=[dem,
                                                                              slope])
            p.extract_sample()


def build_model(path, model_path):
    first = True
    for key, obj in OBJECT_MAP.items():
        root = os.path.join(ROOT, key)
        geo = obj(root)
        pkl_data = PixelTrainingArray(root=root, geography=geo, from_pkl=True)
        if first:
            training_data = {'data': pkl_data.data, 'target_values': pkl_data.target_values,
                             'features': pkl_data.features, 'model_map': pkl_data.model_map,
                             }
            first = False
        else:
            training_data = concatenate_training_data(training_data, pkl_data)

    p = PixelTrainingArray(from_dict=training_data)
    p.to_pickle(training_data, path)
    model_path = mlp(p, model_path)

    for key, obj in OBJECT_MAP.items():
        dst = os.path.join(ROOT, key, 'classified_rasters')
        if not os.path.isdir(dst):
            os.mkdir(dst)

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


class ArrayDisAssembly(object):

    def __init__(self, arr):
        self.arrays = None
        self.n_sections = None
        self.assembled = None
        self.axis = None

        if isinstance(arr, list):
            self.arrays = arr
            self.assembled = self.assemble(arr)

        self.original = arr
        self.shape = arr.shape

    def disassemble(self, n_sections, axis=1):
        self.arrays = array_split(self.original, n_sections, axis=axis)

        self.n_sections = n_sections
        return self.arrays

    def assemble(self, *arrs):
        self.assembled = concatenate((x for x in arrs), axis=0)
        return self.assembled


def get_classifier(obj):
    return obj.classify()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    # build_training_feature_array(skip_landsat=True)

    data_path = os.path.join(abspath, 'model_data', 'data.pkl')
    model = os.path.join(abspath, 'model_data', 'model.ckpt')
    # model = build_model(data_path, model)

    array_file = data_path.replace('data.pkl', 'cut_array.pkl')
    d = Classifier()
    d.get_stack(data_path, saved=array_file)
    data = d.masked_data_stack
    raster_metadata = d.raster_geo
    d = None

    cores = cpu_count()
    a = ArrayDisAssembly(data)
    arrays = a.disassemble(n_sections=cores)
    classifiers = [Classifier(idx=i, arr=a, model=model) for i, a in enumerate(arrays)]
    pool = Pool(processes=cores)

    with pool as p:
        print('running pool on {} objects'.format(len(classifiers)))
        results = [p.apply_async(get_classifier(c), ()) for c in classifiers]
        print('running get')
        classified_arrays = [res.get() for res in results]
        print(classified_arrays)
    pass
# ========================= EOF ====================================================================
