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
import numpy as np
import tensorflow as tf
from pandas import get_dummies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


def neural_net(data):
    """
    :param data: Use the prep_structured_data.StructuredData class.
    :return:
    """

    N = len(data.classes)
    x = data.x
    x = minmax_scale(x)
    m = data.x.shape[0]
    n = data.x.shape[1]
    y = data.y
    nodes = 500
    eta = 0.05
    epochs = 10000
    seed = 128

    x, x_test, y, y_test = train_test_split(x, y, test_size=0.50,
                                            random_state=None)
    x_test, x_validate, y_test, y_validate = train_test_split(x_test, y_test,
                                                              test_size=0.50,
                                                              random_state=None)

    y = get_dummies(y).values
    y_validate = get_dummies(y_validate).values
    y_test = get_dummies(y_test).values

    batch_size = int(np.floor(x.shape[0] / 10))
    graph = tf.Graph()

    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, n))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, N))
        tf_valid_dataset = tf.constant(x_validate)
        tf_test_dataset = tf.constant(x_test)

        weights = {
            'hidden': tf.Variable(tf.random_normal([n, nodes], seed=seed)),
            'output': tf.Variable(tf.random_normal([nodes, N], seed=seed))}
        biases = {
            'hidden': tf.Variable(tf.random_normal([nodes], seed=seed)),
            'output': tf.Variable(tf.random_normal([N], seed=seed))}

        hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
        hidden_layer = tf.nn.relu(hidden_layer)
        output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,
                                                                      labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost)

        train_prediction = tf.nn.softmax(output_layer)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights['output']) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights['output']) + biases)

    with tf.Session(graph=graph) as sess:
        # initialize weights and biases
        tf.global_variables_initializer().run()
        print("Initialized")

        for step in range(epochs):
            # pick a randomized offset
            offset = np.random.randint(0, y.shape[0] - batch_size - 1)

            # Generate a minibatch.
            batch_data = x[offset:(offset + batch_size), :]
            batch_labels = y[offset:(offset + batch_size), :]

            # Prepare the feed dict
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_labels: batch_labels}

            # run one step of computation
            _, l, predictions = sess.run([optimizer, cost, train_prediction],
                                         feed_dict=feed_dict)

            if step % 1000 == 0:
                print("Minibatch loss at step {0}: {1}".format(step, l))
                print("Minibatch accuracy: {:.1f}%".format(
                    accuracy(predictions, batch_labels)))
                print("Validation accuracy: {:.1f}%".format(
                    accuracy(valid_prediction.eval(), y_validate)))

        print("\nTest accuracy: {:.1f}%".format(
            accuracy(test_prediction.eval(), y_test)))


def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu


if __name__ == '__main__':
    home = os.path.expanduser('~')

# ========================= EOF ====================================================================
