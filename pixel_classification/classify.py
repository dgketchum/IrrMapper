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
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
from pickle import load, dump
from numpy import zeros, array, float16, ndarray
from numpy.ma import array as marray
from sklearn.preprocessing import StandardScaler
from rasterio import open as rasopen
from rasterio.dtypes import float32
from rasterio.errors import RasterioIOError

from pixel_classification.compose_array import PixelTrainingArray
from pixel_classification.tf_multilayer_perceptron import multilayer_perceptron


class Result:

    def __init__(self, idx, arr):
        self.idx = idx
        self.arr = arr


class Classifier(object):

    def __init__(self, idx=None, arr=None, model=None):
        self.idx = idx
        self.sess = None
        self.model = None
        self.classifier = None
        self.pixel = None
        self.new_array = None
        self.raster_geo = None
        self.n = None
        self.final_shape = None
        self.masked_data_stack = None

        if isinstance(arr, ndarray):
            self.masked_data_stack = arr
            self.n = self.masked_data_stack.shape[0]

        if model:
            self.model = model
            # self.load_model()

    def get_stack(self, path, saved=None, outfile=None):

        first = True

        if saved:
            stack = load(open(saved, 'rb'))

        data = PixelTrainingArray()
        data.from_pickle(path)

        for i, feat in enumerate(data.features):
            feature_raster = data.model_map[feat]
            try:
                with rasopen(feature_raster, mode='r') as src:
                    arr = src.read()
                    self.raster_geo = src.meta.copy()
                    if saved:
                        break
            except RasterioIOError:
                feature_raster = feature_raster.replace('dgketchum', 'david.ketchum')
                with rasopen(feature_raster, mode='r') as src:
                    arr = src.read()
                    self.raster_geo = src.meta.copy()
            if first:
                empty = zeros((len(data.model_map.keys()), arr.shape[1], arr.shape[2]), float16)
                stack = empty
                stack[i, :, :] = self.normalize_image_channel(arr)
                first = False
            else:
                stack[i, :, :] = self.normalize_image_channel(arr)

        if outfile:
            with open(outfile, 'wb') as handle:
                dump(stack, handle, protocol=4)

        self.final_shape = 1, stack.shape[1], stack.shape[2]
        stack = stack.reshape((stack.shape[0], stack.shape[1] * stack.shape[2]))
        stack[stack == 0.] = np.nan
        self.masked_data_stack = marray(stack, mask=np.isnan(stack))
        self.n = self.masked_data_stack.shape[0]
        del stack

        self.new_array = np.zeros_like(arr.reshape((1, arr.shape[1] * arr.shape[2])), dtype=float16)

    def classify(self, arr=None):

        sess = tf.Session()
        saver = tf.train.import_meta_graph('{}.meta'.format(self.model))
        saver.restore(sess, self.model)
        self.pixel = tf.placeholder("float", [None, self.n])

        wh = sess.graph.get_tensor_by_name('Wh:0')
        wo = sess.graph.get_tensor_by_name('Wo:0')
        bh = sess.graph.get_tensor_by_name('Bh:0')
        bo = sess.graph.get_tensor_by_name('Bo:0')
        classifier = tf.add(tf.matmul(multilayer_perceptron(self.pixel, wh, bh), wo), bo)

        if isinstance(arr, ndarray):
            if len(arr.shape) > 2:
                self.masked_data_stack = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
            elif len(arr.shape) == 2:
                self.masked_data_stack = arr
            else:
                raise AttributeError('Invalid shape')

        g = tf.get_default_graph()

        ct_out = 0
        ct_nan = 0

        if not self.new_array:
            self.new_array = zeros((1, self.masked_data_stack.shape[1]), dtype=float16)

        for i in range(self.masked_data_stack.shape[-1]):
            if not np.ma.is_masked(self.masked_data_stack[:, i]):
                dat = self.masked_data_stack[:, i]
                dat = array(dat).reshape((1, dat.shape[0]))
                loss = sess.run(classifier, feed_dict={self.pixel: dat})
                self.new_array[0, i] = np.argmax(loss, 1)
                ct_out += 1
            else:
                self.new_array[0, i] = np.nan
                ct_nan += 1

        sess.close()

        self.new_array = array(self.new_array, dtype=float32)

        return Result(self.idx, self.new_array)

    def write_raster(self, out_location, out_name, new_array=None):

        if isinstance(new_array, ndarray):
            self.new_array = new_array

        try:
            self.new_array = self.new_array.reshape(1, self.new_array.shape[1],
                                                    self.new_array.shape[2])
        except IndexError:
            self.new_array = self.new_array.reshape(1, self.new_array.shape[0],
                                                    self.new_array.shape[1])

        self.raster_geo['dtype'] = str(array.dtype)
        self.raster_geo['count'] = 1

        out_ras = os.path.join(out_location, out_name)

        with rasopen(out_ras, 'w', **self.raster_geo) as dst:
            dst.write(self.new_array)

        return None

    @staticmethod
    def normalize_image_channel(data):
        data = data.reshape((data.shape[1], data.shape[2]))
        scaler = StandardScaler()
        scaler = scaler.fit(data)
        data = scaler.transform(data)
        data = data.reshape((1, data.shape[0], data.shape[1]))
        return data


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
