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
    # s = '1553014193.4813933'
    # f = 'training_data/multiclass/train/class_2_data/{}.pkl'.format(s)
    # with open(f, 'rb') as f:
    #     data = pload(f)
    # data = np.expand_dims(data['data'], axis=0)
    # data = np.swapaxes(data, 1, 3)
    # gradient_wrt_inputs(model, data)
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


def unet_same_padding(input_shape, weight_shape, initial_exp=6, n_classes=5):
     
    features = Input(shape=input_shape)
    weights = Input(shape=weight_shape)
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

    last_conv = Conv2D(filters=n_classes, kernel_size=1, padding='same', activation='softmax')(c13)
    last = Lambda(lambda x: x / tf.reduce_sum(x, len(x.get_shape()) - 1, True))(last_conv)
    last = Lambda(lambda x: tf.clip_by_value(x, _epsilon, 1. - _epsilon))(last)
    last = Lambda(lambda x: K.log(x))(last)
    weighted_xen = multiply([last, weights])
    return Model(inputs=[features, weights], outputs=[weighted_xen])


def unet_valid_padding(input_shape, weighted_input_shape, n_classes, base_exp=5):
    ''' 
    This model does not use any Conv2DTranspose layers. 
    Instead a Upsampling2D layer with a Conv layer after 
    with same padding. 
    '''
    inp1 = Input(input_shape)
    weighted_input = Input(shape=weighted_input_shape)
    base = 2
    exp = base_exp

    # 64 filters
    c1 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(inp1)
    c2 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c1)
    c2 = BatchNormalization()(c2)
    mp1 = MaxPooling2D(pool_size=2, strides=(2, 2))(c2)

    exp += 1
    # 128 filters
    c3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(mp1)
    c4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c3)
    c4 = BatchNormalization()(c4)
    mp2 = MaxPooling2D(pool_size=2, strides=(2, 2))(c4)


    exp += 1
    # 256 filters
    c5 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(mp2)
    c6 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c5)
    c6 = BatchNormalization()(c6)
    mp3 = MaxPooling2D(pool_size=2, strides=(2, 2))(c6)

    exp += 1
    # 512 filters
    c7 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(mp3)
    c8 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c7)
    c8 = BatchNormalization()(c8)

    mp4 = MaxPooling2D(pool_size=2, strides=(2, 2))(c8)

    exp += 1
    # 1024 filters
    c9 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(mp4)
    c10 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c9)
    c10 = BatchNormalization()(c10)

    exp -= 1
    # 512 filters, making 1024 when concatenated with 
    # the corresponding layer from the contracting path.
    # u1 = Conv2DTranspose(filters=base**exp, strides=(2, 2), kernel_size=(2, 2),
    #         activation='relu')(c10)
    u1 = UpSampling2D(size=(2, 2))(c10)
    u1 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u1)

    c8_cropped = Cropping2D(cropping=4)(c8)
    concat_u1_c8 = Concatenate()([u1, c8_cropped])

    # 512 filters
    c11 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', 
            padding='valid')(concat_u1_c8)

    exp -= 1
    # 256 filters, making 512 when concatenated with the 
    # corresponding layer from the contracting path.
    c12 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c11)

    # u2 = Conv2DTranspose(filters=base**exp, strides=(2, 2), kernel_size=(2, 2),
    #         activation='relu')(c12)
    u2 = UpSampling2D(size=(2, 2))(c12)
    u2 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u2)

    c6_cropped = Cropping2D(cropping=16)(c6)
    concat_u2_c6 = Concatenate()([u2, c6_cropped])

    # 256 filters
    c13 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', 
            padding='valid')(concat_u2_c6)
    bn1 = BatchNormalization()(c13)

    exp -= 1
    # 128 filters, making 256 when concatenated with the 
    # corresponding layer from the contracting path.
    c14 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(bn1)

    # u3 = Conv2DTranspose(filters=base**exp, strides=(2, 2), kernel_size=(2, 2),
    #         activation='relu')(c14)
    u3 = UpSampling2D(size=(2, 2))(c14)
    u3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u3)

    c4_cropped = Cropping2D(cropping=40)(c4)
    concat_u3_c4 = Concatenate()([u3, c4_cropped])

    # 128 filters
    c15 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', 
            padding='valid')(concat_u3_c4)
    bn2 = BatchNormalization()(c15)

    exp -= 1
    # 64 filters, making 128 when concatenated with the 
    # corresponding layer from the contracting path.
    c16 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(bn2)

    #u4 = Conv2DTranspose(filters=base**exp, strides=(2, 2), kernel_size=(2, 2),
    #        activation='relu')(c16)
    u4 = UpSampling2D(size=(2, 2))(c16)
    u4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u4)

    c2_cropped = Cropping2D(cropping=88)(c2)
    concat_u4_c2 = Concatenate()([u4, c2_cropped])

    c17 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu',
            padding='valid')(concat_u4_c2)
    bn3 = BatchNormalization()(c17)

    c18 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu',
            padding='valid')(bn3)
    c18 = BatchNormalization()(c18)

    last_conv = Conv2D(filters=n_classes, kernel_size=1, activation='softmax', padding='valid')(c18)

    last = Lambda(lambda x: x / tf.reduce_sum(x, len(x.get_shape()) - 1, True))(last_conv)
    last = Lambda(lambda x: tf.clip_by_value(x, _epsilon, 1. - _epsilon))(last)
    last = Lambda(lambda x: K.log(x))(last)
    weighted_sum = multiply([last, weighted_input])
    return Model(inputs=[inp1, weighted_input], outputs=[weighted_sum])


def unet(n_classes, channel_depth=36):
    x = Input((None, None, channel_depth))
    base = 2
    exp = 5

    # 64 filters
    c1 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(x)
    c2 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c1)
    mp1 = MaxPooling2D(pool_size=2, strides=(2, 2))(c2)

    exp += 1
    # 128 filters
    c3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(mp1)
    c4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c3)
    mp2 = MaxPooling2D(pool_size=2, strides=(2, 2))(c4)


    exp += 1
    # 256 filters
    c5 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(mp2)
    c6 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c5)
    mp3 = MaxPooling2D(pool_size=2, strides=(2, 2))(c6)

    exp += 1
    # 512 filters
    c7 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(mp3)
    c8 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c7)
    mp4 = MaxPooling2D(pool_size=2, strides=(2, 2))(c8)

    exp += 1
    # 1024 filters
    c9 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(mp4)
    c10 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c9)

    exp -= 1
    # 512 filters, making 1024 when concatenated with 
    # the corresponding layer from the contracting path.
    u1 = Conv2DTranspose(filters=base**exp, strides=(2, 2), kernel_size=(2, 2),
            activation='relu')(c10)

    c8_cropped = Cropping2D(cropping=4)(c8)
    concat_u1_c8 = Concatenate()([u1, c8_cropped])

    # 512 filters
    c11 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', 
            padding='valid')(concat_u1_c8)

    exp -= 1
    # 256 filters, making 512 when concatenated with the 
    # corresponding layer from the contracting path.
    c12 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(c11)

    u2 = Conv2DTranspose(filters=base**exp, strides=(2, 2), kernel_size=(2, 2),
            activation='relu')(c12)

    c6_cropped = Cropping2D(cropping=16)(c6)
    concat_u2_c6 = Concatenate()([u2, c6_cropped])

    # 256 filters
    c13 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', 
            padding='valid')(concat_u2_c6)
    bn1 = BatchNormalization(axis=3)(c13)

    exp -= 1
    # 128 filters, making 256 when concatenated with the 
    # corresponding layer from the contracting path.
    c14 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(bn1)

    u3 = Conv2DTranspose(filters=base**exp, strides=(2, 2), kernel_size=(2, 2),
            activation='relu')(c14)

    c4_cropped = Cropping2D(cropping=40)(c4)
    concat_u3_c4 = Concatenate()([u3, c4_cropped])

    # 128 filters
    c15 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', 
            padding='valid')(concat_u3_c4)
    bn2 = BatchNormalization(axis=3)(c15)

    exp -= 1
    # 64 filters, making 128 when concatenated with the 
    # corresponding layer from the contracting path.
    c16 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(bn2)

    u4 = Conv2DTranspose(filters=base**exp, strides=(2, 2), kernel_size=(2, 2),
            activation='relu')(c16)

    c2_cropped = Cropping2D(cropping=88)(c2)
    concat_u4_c2 = Concatenate()([u4, c2_cropped])

    c17 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu',
            padding='valid')(concat_u4_c2)
    bn3 = BatchNormalization(axis=3)(c17)

    c18 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu',
            padding='valid')(bn3)

    last = Conv2D(filters=n_classes, kernel_size=1, activation='linear', padding='valid')(c18)
    return Model(inputs=x, outputs=last)


