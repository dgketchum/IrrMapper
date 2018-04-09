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
from numpy import unique
from numpy.random import randint
import tensorflow as tf
from pandas import get_dummies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pixel_classification.compose_array import PixelTrainingArray


def mlp(data):
    """
    :param data: Use the PixelTrainingArray class.
    :return:
    """
    assert isinstance(data, PixelTrainingArray)

    x = normalize(data.data)
    y = get_dummies(data.target_values).values
    N = len(unique(data.features))
    n = data.data.shape[1]

    nodes = 500
    eta = 0.05
    epochs = 1000
    seed = 128

    x, x_test, y, y_test = train_test_split(x, y, test_size=0.33,
                                            random_state=None)

    X = tf.placeholder("float", [None, n])
    Y = tf.placeholder("float", [None, N])

    batch_size = 100

    weights = {
        'hidden': tf.Variable(tf.random_normal([n, nodes], seed=seed)),
        'output': tf.Variable(tf.random_normal([nodes, N], seed=seed))}
    biases = {
        'hidden': tf.Variable(tf.random_normal([nodes], seed=seed)),
        'output': tf.Variable(tf.random_normal([N], seed=seed))}

    y_pred = tf.add(tf.matmul(multilayer_perceptron(X, weights['hidden'], biases['hidden']),
                              weights['output']), biases['output'])

    loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss_op)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for step in range(epochs):
        offset = randint(0, data.y.shape[0] - batch_size - 1)

        batch_data = data.x[offset:(offset + batch_size), :]
        batch_labels = data.y[offset:(offset + batch_size), :]

        feed_dict = {X: batch_data, Y: batch_labels}

        _, loss = sess.run([optimizer, loss_op],
                           feed_dict=feed_dict)

        if step % 100 == 0:
            pred = tf.nn.softmax(y_pred)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('Test accuracy: {}, loss {}'.format(accuracy.eval({X: x_test, Y: y_test}), loss))


def multilayer_perceptron(x, weights, biases):
    out_layer = tf.add(tf.matmul(x, weights), biases)
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


def normalize(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    return data


if __name__ == '__main__':
    home = os.path.expanduser('~')

# ========================= EOF ====================================================================
