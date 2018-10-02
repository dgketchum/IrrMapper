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
from copy import deepcopy
from datetime import datetime
from multiprocessing import cpu_count

from multiprocess.pool import Pool
from numpy.core.multiarray import concatenate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
from numpy import load, save
from numpy import zeros, array, float16, ndarray, array_split, float64
from numpy.ma import array as marray
from sklearn.preprocessing import StandardScaler
from rasterio import open as rasopen
from rasterio.dtypes import float32

from sat_image.warped_vrt import warp_single_image
from pixel_classification.prepare_images import ImageStack
from pixel_classification.tf_multilayer_perceptron import multilayer_perceptron

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


class Result:

    def __init__(self, idx, arr):
        self.idx = idx
        self.arr = arr


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
        self.feature_ras = None
        self.data = None
        self.saved_array = None
        self.mask = None

        if isinstance(arr, ndarray):
            self.masked_data_stack = arr
            self.n = self.masked_data_stack.shape[0]

        if model:
            self.model = model

    def get_stack(self, image_data, outfile=None, mask_path=None):

        if mask_path:
            self.mask = self._get_mask_from_raster(mask_path)

        if isinstance(image_data, str):
            print('load {}'.format(image_data))
            self.saved_array = image_data
            stack = load(image_data)

        elif isinstance(image_data, ImageStack):
            self.data = image_data
            stack = self._get_stack_channels()

        else:
            raise NotImplementedError('Must pass ImageStack or path to saved array (.npy)')

        if outfile:
            print('saving image stack {}'.format(outfile))
            save(outfile, stack)

        self.final_shape = 1, stack.shape[1], stack.shape[2]
        stack = stack.reshape((stack.shape[0], stack.shape[1] * stack.shape[2]))
        stack[stack == 0.] = np.nan

        if mask_path:
            ms = self.mask.shape
            msk = np.repeat(self.mask.reshape((ms[0], ms[1] * ms[2])), stack.shape[0], axis=0)
            stack = marray(stack, mask=msk)

        self.masked_data_stack = marray(stack, mask=np.isnan(stack))
        self.n = self.masked_data_stack.shape[0]
        del stack

        self.new_array = zeros((1, self.masked_data_stack.shape[1]), dtype=float16)

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

        if not self.new_array:
            self.new_array = zeros((1, self.masked_data_stack.shape[1]), dtype=float16)

        for i in range(self.masked_data_stack.shape[-1]):
            if not np.ma.is_masked(self.masked_data_stack[:, i]):
                dat = self.masked_data_stack[:, i]
                dat = array(dat).reshape((1, dat.shape[0]))
                loss = sess.run(classifier, feed_dict={self.pixel: dat})
                self.new_array[0, i] = np.argmax(loss, 1)
            else:
                self.new_array[0, i] = np.nan

        sess.close()

        self.new_array = array(self.new_array, dtype=float32)

        return Result(self.idx, self.new_array)

    def write_raster(self, out_file, new_array=None):

        if isinstance(new_array, ndarray):
            self.new_array = new_array

        try:
            self.new_array = self.new_array.reshape(1, self.new_array.shape[1],
                                                    self.new_array.shape[2])
        except IndexError:
            self.new_array = self.new_array.reshape(1, self.new_array.shape[0],
                                                    self.new_array.shape[1])

        self.raster_geo['dtype'] = str(self.new_array.dtype)
        self.raster_geo['count'] = 1

        with rasopen(out_file, 'w', **self.raster_geo) as dst:
            dst.write(self.new_array)

        return None

    def _get_mask_from_raster(self, extra_mask):
        with rasopen(extra_mask, mode='r') as src:
            arr = src.read()
            self.raster_geo = src.meta.copy()
        return arr

    def _get_stack_channels(self):

        stack = None
        first = True

        for i, feat in enumerate(self.data.paths_map.keys()):

            self.feature_ras = self.data.paths_map[feat]

            with rasopen(self.feature_ras, mode='r') as src:
                arr = src.read()
                self.raster_geo = src.meta.copy()
                if self.saved_array:
                    break
            if first:
                first_geo = deepcopy(self.raster_geo)
                empty = zeros((len(self.data.paths_map.keys()), arr.shape[1], arr.shape[2]), float16)
                stack = empty
                stack[i, :, :] = self.normalize_image_channel(arr)
                first = False
            else:
                try:
                    stack[i, :, :] = self.normalize_image_channel(arr)
                except ValueError:
                    arr = warp_single_image(self.feature_ras, first_geo)
                    stack[i, :, :] = self.normalize_image_channel(arr)

        return stack

    @staticmethod
    def normalize_image_channel(data):
        data = data.reshape((data.shape[1], data.shape[2]))

        data[data == np.nan] = 0.
        data[data == np.inf] = 0.
        scaler = StandardScaler()

        try:
            scaler = scaler.fit(data)
            data = scaler.transform(data)
        except ValueError:
            # this handles large float arrays
            data = data.astype(dtype=float64)
            data = (data - np.nanmean(data)) / np.nanstd(data)

        data = data.reshape((1, data.shape[0], data.shape[1]))
        data = data.astype(dtype=float16)
        return data


def get_classifier(obj, arr):
    return obj.classify(arr)


def classify_multiproc(model, stack_data, result, array_outfile=None, mask=None):
    d = Classifier()
    d.get_stack(stack_data, outfile=array_outfile, mask_path=mask)
    stack_data = d.masked_data_stack

    cores = cpu_count()
    a = ArrayDisAssembly(stack_data)
    arrays = a.disassemble(n_sections=cores)
    classifiers = [Classifier(idx=i, arr=a, model=model) for i, a in enumerate(arrays)]
    pool = Pool(processes=cores)
    time = datetime.now()
    print('async')
    with pool as p:
        pool_results = [p.apply_async(get_classifier, (c, a)) for a, c in zip(arrays, classifiers)]
        classified_arrays = [res.get() for res in pool_results]
        a.assemble(classified_arrays)
        final = a.assembled.reshape(d.final_shape)
    td = (datetime.now() - time)

    print('time', td.days, td.seconds // 3600, (td.seconds // 60) % 60)

    d.write_raster(out_file=result, new_array=final)

    return None


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
