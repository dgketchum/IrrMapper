# =============================================================================================
# Copyright 2017 dgketchum
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
# =============================================================================================

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import scale, normalize
# from sklearn.metrics import confusion_matrix, accuracy_score


def softmax(data):
    '''
    :param data: Use the prep_structured_data.StructuredData class.
    :return:
    '''
    classes = data.classes
    x = data.x
    m = data.x.shape[0]
    n = data.x.shape[1]
    y = data.y
    eta = 0.05
    batch_size = m / 10.
    epochs = 10000

    d, d_test, y, y_test = train_test_split(x, y, test_size=0.50,
                                            random_state=None)
    d_test, d_validate, y_test, y_validate = train_test_split(d_test, y_test, test_size=0.50,
                                                              random_state=None)

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, m))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n))
        tf_valid_dataset = tf.constant(d_validate)
        tf_test_dataset = tf.constant(d_test)

        weights = tf.Variable(tf.truncated_normal([m, n]))
        biases = tf.Variable(tf.zeros([n]))

        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf_train_labels, logits=logits))

        optimizer = tf.train.GradientDescentOptimizer(eta).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

        pass


if __name__ == '__main__':
    home = os.path.expanduser('~')


# ========================= EOF ================================================================
