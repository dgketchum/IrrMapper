import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (multiply, Conv2D, Input, MaxPooling2D, Conv2DTranspose,
        Concatenate, Dropout, UpSampling2D, BatchNormalization, Cropping2D, Lambda, Activation)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu


def gradient_wrt_inputs(model, data):
    layer_output = model.output
    loss = -tf.reduce_mean(layer_output)
    grads = K.gradients(loss, model.input[0])[0]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    weights = np.ones((1, 388, 388, 5))
    results = sess.run(grads, feed_dict={model.input[0]:data, model.input[1]:weights})
    return results


_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)


def ConvBlock(x, filters=64):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
        kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    return Activation(relu)(x)

def ConvBNRelu(x, filters=64):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    return Activation(relu)(x)


_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)


def unet(input_shape, initial_exp=6, n_classes=5):
     
    features = Input(shape=input_shape)
    _power = initial_exp
    exp = 2

    c1 = ConvBlock(features, exp**_power)
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
    c5 = ConvBlock(mp4, exp**_power)
    _power -= 1

    u1 = UpSampling2D(size=(2, 2))(c5)
    c6 = ConvBNRelu(u1, filters=exp**_power)
    u1_c4 = Concatenate()([c6, c4])
    c7 = ConvBlock(u1_c4, filters=exp**_power)

    _power -= 1
    
    u2 = UpSampling2D(size=(2, 2))(c7)
    c8 = ConvBNRelu(u2, filters=exp**_power)
    u2_c3 = Concatenate()([c8, c3])
    c9 = ConvBlock(u2_c3, filters=exp**_power)

    _power -= 1
    
    u3 = UpSampling2D(size=(2, 2))(c9)
    c10 = ConvBNRelu(u3, filters=exp**_power)
    u3_c2 = Concatenate()([c10, c2])
    c11 = ConvBlock(u3_c2, filters=exp**_power)

    _power -= 1
    u4 = UpSampling2D(size=(2, 2))(c11)
    c12 = ConvBNRelu(u4, filters=exp**_power)
    u4_c1 = Concatenate()([c12, c1])
    c13 = ConvBlock(u4_c1, filters=exp**_power)


    logits = Conv2D(filters=n_classes, kernel_size=1, strides=1,
                    activation=None, name='logits')(c13)
    
    return Model(inputs=[features], outputs=[logits])

if __name__ == '__main__':
    pass
