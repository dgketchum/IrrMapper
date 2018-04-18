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
from datetime import datetime
import numpy as np
from numpy import unique
from numpy import zeros, float16, array
from numpy.random import randint
from numpy.ma import array as marray
import tensorflow as tf
from pandas import get_dummies
from rasterio import open as rasopen
from rasterio.dtypes import float32
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def mlp(data, checkpoint=None):
    """
    :param checkpoint:
    :param data: Use the PixelTrainingArray class.
    :return:
    """

    x = normalize_feature_array(data.data)
    y = get_dummies(data.target_values).values
    N = len(unique(data.target_values))
    n = data.data.shape[1]

    nodes = 300
    eta = 0.01
    epochs = 10000
    seed = 128
    batch_size = 500

    x, x_test, y, y_test = train_test_split(x, y, test_size=0.33,
                                            random_state=None)

    X = tf.placeholder("float", [None, n])
    Y = tf.placeholder("float", [None, N])

    weights = {
        'hidden': tf.Variable(tf.random_normal([n, nodes], seed=seed), name='Wh'),
        'output': tf.Variable(tf.random_normal([nodes, N], seed=seed), name='Wo')}
    biases = {
        'hidden': tf.Variable(tf.random_normal([nodes], seed=seed), name='Bh'),
        'output': tf.Variable(tf.random_normal([N], seed=seed), name='Bo')}

    y_pred = tf.add(tf.matmul(multilayer_perceptron(X, weights['hidden'], biases['hidden']),
                              weights['output']), biases['output'])

    loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss_op)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for step in range(epochs):

        offset = randint(0, y.shape[0] - batch_size - 1)

        batch_data = x[offset:(offset + batch_size), :]
        batch_labels = y[offset:(offset + batch_size), :]

        feed_dict = {X: batch_data, Y: batch_labels}

        _, loss = sess.run([optimizer, loss_op],
                           feed_dict=feed_dict)

        if step % 1000 == 0:
            pred = tf.nn.softmax(y_pred)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('Test accuracy: {}, loss {}'.format(accuracy.eval({X: x_test, Y: y_test}), loss))

    features = data.features.tolist()
    stack = None
    first = True
    arr = None
    for i, feat in enumerate(features):
        with rasopen(data.model_map[feat], mode='r') as src:
            arr = src.read()
            meta = src.meta.copy()
        if first:
            empty = zeros((len(features), arr.shape[1], arr.shape[2]), float16)
            stack = empty
            stack[i, :, :] = normalize_image_channel(arr)

            first = False
        else:
            stack[i, :, :] = normalize_image_channel(arr)

    final_shape = 1, stack.shape[1], stack.shape[2]
    stack = stack.reshape((stack.shape[0], stack.shape[1] * stack.shape[2]))
    stack[stack == 0.] = np.nan
    m_stack = marray(stack, mask=np.isnan(stack))

    new_array = np.zeros_like(arr.reshape((1, arr.shape[1] * arr.shape[2])), dtype=float16)

    pixel = tf.placeholder("float", [None, n])
    classify = tf.add(tf.matmul(multilayer_perceptron(pixel, weights['hidden'], biases['hidden']),
                                weights['output']), biases['output'])
    time = datetime.now()

    ct_nan = 0
    ct_out = 0

    for i in range(m_stack.shape[-1]):
        if not np.ma.is_masked(m_stack[:, i]):
            dat = m_stack[:, i]
            dat = array(dat).reshape((1, dat.shape[0]))
            loss = sess.run(classify, feed_dict={pixel: dat})
            new_array[0, i] = np.argmax(loss, 1)
            ct_out += 1
        else:
            new_array[0, i] = np.nan
            ct_nan += 1

        if i % 1000000 == 0:
            print('Count {} of {} pixels in {} seconds'.format(i, m_stack.shape[-1],
                                                               (datetime.now() - time).seconds))

    new_array = new_array.reshape(final_shape)
    new_array = array(new_array, dtype=float32)

    meta['count'] = 1
    meta['dtype'] = float32
    with rasopen(os.path.join(checkpoint, 'binary_raster.tif'), 'w', **meta) as dst:
        dst.write(new_array)
    return None


def multilayer_perceptron(x, weights, biases):
    out_layer = tf.add(tf.matmul(x, weights), biases)
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


def normalize_feature_array(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    return data


def normalize_image_channel(data):
    data = data.reshape((data.shape[1], data.shape[2]))
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    data = data.reshape((1, data.shape[0], data.shape[1]))
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


if __name__ == '__main__':
    home = os.path.expanduser('~')

# ========================= EOF ====================================================================
