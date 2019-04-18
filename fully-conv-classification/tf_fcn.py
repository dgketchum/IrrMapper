import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (Activation, Conv2D, UpSampling2D, BatchNormalization, MaxPooling2D, Input, Concatenate, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import l2
from tensorflow.data import Dataset
from data_generators import generate_unbalanced_data

import tensorflow.keras.backend as K

def ConvBlock(x, filters=64, expanding_path=False):

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    if expanding_path:
        x = Conv2D(filters=filters // 2, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(x)
    else:
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    return Activation(relu)(x)


_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)

def model_func(input_shape, initial_exp=6, n_classes=5):
     
    inp = Input(shape=input_shape)
    _power = initial_exp
    exp = 2

    c1 = ConvBlock(inp, exp**_power)
    mp1 = MaxPooling2D(pool_size=2, strides=2)(c1)

    _power += 1

    c2 = ConvBlock(mp1, exp**_power)
    mp2 = MaxPooling2D(pool_size=2, strides=2)(c2)

    _power += 1

    c3 = ConvBlock(mp2, exp**_power)
    mp3 = MaxPooling2D(pool_size=2, strides=2)(c3)

    _power += 1 

    c4 = ConvBlock(mp3, exp**_power)
    mp4 = MaxPooling2D(pool_size=2, strides=2)(c4)

    _power += 1

    # 1024 filters
    c5 = Conv2D(filters=exp**_power, kernel_size=3, strides=1, padding='same')(mp4)
    _power -= 1
    c5 = Conv2D(filters=exp**_power, kernel_size=3, strides=1, padding='same')(c5)

    u1 = UpSampling2D(size=(2, 2))(c5)

    u1_c4 = Concatenate()([u1, c4])

    c6 = ConvBlock(u1_c4, filters=exp**_power, expanding_path=True)

    u2 = UpSampling2D(size=(2, 2))(c6)

    u2_c3 = Concatenate()([u2, c3])

    _power -= 1 
    c7 = ConvBlock(u2_c3, filters=exp**_power, expanding_path=True)

    u3 = UpSampling2D(size=(2, 2))(c7)

    u3_c2 = Concatenate()([u3, c2])

    _power -= 1 
    c8 = ConvBlock(u3_c2, filters=exp**_power, expanding_path=True)

    u4 = UpSampling2D(size=(2, 2))(c8)

    u4_c1 = Concatenate()([u4, c1])

    _power -= 1 
    c9 = ConvBlock(u4_c1, filters=exp**_power)
    last_conv = Conv2D(filters=n_classes, kernel_size=1, padding='same', activation=None)(c9)
    return Model(inputs=[inp], outputs=[last_conv])


def weighted_loss(y_true, logits, weights):
    ''' y_true: one-hot encoding of labels.
        y_pred: tensor of probabilities.
        weights: tensor of weights, 0 where there isn't data.
    Recall:
    L = a0*CE | focal: L = a0*(1-pt)^gamma*CE
    '''
    unweighted_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
    weighted_loss = unweighted_loss*weights
    mask = tf.not_equal(weights, 0)
    weighted_loss = tf.boolean_mask(weighted_loss, mask)
    return tf.reduce_mean(weighted_loss)

def accuracy(y_true, logits):

    mask = tf.not_equal(tf.sum(y_true, axis=len(y_true.get_shape())-1), 0)
    y_pred = tf.nn.softmax(logits)
    y_pred = tf.math.argmax(y_pred)
    y_true = tf.math.argmax(y_true)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return tf.reduce_mean(tf.equal(y_true, y_pred))


input_shape = (None, None, 51)
learning_rate = 1e-3
epochs = 1
model = model_func(input_shape, n_classes=5)
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
loss_fn = weighted_loss
training_directory = 'training_data/train/'
class_weights = {0:4.5, 1:1.0, 2:2.96, 3:14.972, 4:10}
train_data = generate_unbalanced_data(training_directory, class_weights=class_weights)
#train_data = Dataset(train_data)
loss_metric = tf.keras.metrics.Mean(name='train_loss')
acc_metric = tf.keras.metrics.Mean(name='acc')

#@tf.function
def train_step(inputs, labels, weights):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        acc = accuracy(labels, logits)
        reg_loss = tf.math.add_n(model.losses)
        pred_loss = loss_fn(labels, logits, weights)
        total_loss = pred_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_metric.update_state(total_loss)
    acc_metric.update_state(acc)

step = 0
for epoch in range(epochs):
    for inputs, labels, weights in train_data:
        loss = train_step(inputs, labels, weights)
        if step > 100:
            break
