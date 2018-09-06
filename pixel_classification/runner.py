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
from datetime import datetime
from pathos.multiprocessing import cpu_count
from multiprocess.pool import Pool

from pixel_classification.runspec import Montana, Nevada, Oregon, Utah, Washington
from pixel_classification.prepare_images import ImageStack
from pixel_classification.compose_array import PixelTrainingArray
from pixel_classification.tf_multilayer_perceptron import mlp
from pixel_classification.classify import Classifier

home = os.path.expanduser('~')
ROOT = os.path.join(home, 'IrrigationGIS', 'western_states_irrgis')

OBJECT_MAP = {
    'MT': Montana,
    'NV': Nevada,
    'OR': Oregon,
    'UT': Utah,
    'WA': Washington
}


def build_training_feature_array(root):
    for key, obj in OBJECT_MAP.items():
        geography = os.path.join(ROOT, key)
        geo = obj(geography)
        if geo.sat == 8:
            # TODO: add path for multi-scene root, add more parameters to ImageStack instance
            i = ImageStack(root=root, satellite=geo.sat, path=geo.path, row=geo.row,
                           max_cloud_pct=40, year=geo.year)

            p = PixelTrainingArray(root=i.root, geography=geo, instances=5000,
                                   overwrite_array=True, overwrite_points=True,
                                   ancillary_rasters=[dem, slope])

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

    def assemble(self, results, axis=1):
        d = {r.idx: r.arr for r in results}
        l = [d[k] for k in sorted(d.keys())]
        self.assembled = concatenate(l, axis=axis)
        return self.assembled


def get_classifier(obj, arr):
    return obj.classify(arr)


def classify_multiproc(model, data, array):
    d = Classifier()
    d.get_stack(data, saved=array)
    data = d.masked_data_stack

    cores = cpu_count()
    a = ArrayDisAssembly(data)
    arrays = a.disassemble(n_sections=cores)
    classifiers = [Classifier(idx=i, arr=a, model=model) for i, a in enumerate(arrays)]
    pool = Pool(processes=cores)
    time = datetime.now()
    with pool as p:
        pool_results = [p.apply_async(get_classifier, (c, a)) for a, c in zip(arrays, classifiers)]
        classified_arrays = [res.get() for res in pool_results]
        a.assemble(classified_arrays)
        final = a.assembled.reshape(d.final_shape)
    td = (datetime.now() - time)
    print(td.days, td.seconds // 3600, (td.seconds // 60) % 60)
    d.write_raster(out_location=model_data, out_name='test_classified.tif', new_array=final)

    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')

    build_training_feature_array()
    model_data = os.path.join(abspath, 'model_data')
    data_path = os.path.join(model_data, 'data.pkl')
    model = os.path.join(model_data, 'model.ckpt')
    # model = build_model(data_path, model)
    array_file = data_path.replace('data.pkl', 'array.pkl')
    classify_multiproc(model, data_path, array_file)

# ========================= EOF ====================================================================
