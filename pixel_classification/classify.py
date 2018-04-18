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
from numpy import zeros, uint16, where

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from pixel_classification.tf_multilayer_perceptron import mlp
from pixel_classification.tf_softmax import softmax
from pixel_classification.compose_array import PixelTrainingArray


def apply_model(checkpoint, pixel_data, max_memory=2, write_stack=False):
    data = PixelTrainingArray(pickle_path=pixel_data)
    # data_size = get_size(os.path.dirname(pixel_data))

    features = data.features.tolist()
    stack = None
    first = True
    for i, feat in enumerate(features):
        with rasopen(data.model_map[feat], mode='r') as src:
            arr = src.read()
            meta = src.meta.copy()
        if first:
            empty = zeros((len(features), arr.shape[1], arr.shape[2]), uint16)
            stack = empty
            stack[i, :, :] = normalize(arr)
            first = False
            mask = where(arr != 0)
        else:
            stack[i, :, :] = normalize(arr)

    if write_stack:
        meta['count'] = i + 1
        meta['dtype'] = uint16
        with rasopen(pixel_data.replace('data.pkl', 'stack.tif'), 'w', **meta) as dst:
            for i in range(1, stack.shape[0] + 1):
                dst.write(stack[i - 1, :, :], i)

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(os.path.join(checkpoint, 'model.meta'))
        path = tf.train.get_checkpoint_state(checkpoint)
        new_saver.restore(sess, path.model_checkpoint_path)


    pass


def normalize(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    return data


def get_size(start_path='.'):
    """ Size of data directory in GB.
    :param start_path:
    :return:
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    total_size = total_size * 1e-9
    return total_size


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
    root = os.path.dirname(__file__).replace('pixel_classification', 'landsat_data')

    year = os.path.join(root, '39', '27', '2008')
    p_path = os.path.join(year, 'data.pkl')

    # build_model(p_path, alg='mlp', model=model)
    apply_model(year, p_path)

# ========================= EOF ====================================================================
