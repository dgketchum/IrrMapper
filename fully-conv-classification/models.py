import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (multiply, Conv2D, Input, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout, UpSampling2D, BatchNormalization, Cropping2D, Lambda)

def fcnn_model(n_classes):
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=8, padding='same', activation='relu',
        input_shape=(None, None, 36)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=4, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(filters=n_classes, kernel_size=2, padding='same',
        activation='softmax')) 
    #model.summary()
    return model

def fcnn_functional_small(n_classes):
    x = Input((None, None, 36))

    c1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
    c1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(c1)
    mp1 = MaxPooling2D(pool_size=2, strides=(2, 2))(c1)

    c2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(mp1)
    c2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(c2)
    mp2 = MaxPooling2D(pool_size=2, strides=(2, 2))(c2)
    #mp2 = Dropout(0.5)(mp2)

    c3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(mp2)
    c3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(c3)
    mp3 = MaxPooling2D(pool_size=2, strides=(2, 2))(c3)
    
    last_conv = Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(mp3)

    u1 = UpSampling2D(size=(2, 2))(last_conv)
    u1 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(u1)
    u1 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(u1)

    u1_c3 = Concatenate()([c3, u1])

    u2 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(u1_c3)
    u2 = UpSampling2D(size=(2, 2))(u2)
    u2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(u2)
    u2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(u2)
    #u2 = Dropout(0.5)(u2)

    u2_c2 = Concatenate()([u2, c2])
    u2_c2 = Dropout(0.5)(u2_c2)

    c4 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(u2_c2)
    u3 = UpSampling2D(size=(2, 2))(c4)
    u3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(u3)

    u3_c1 = Concatenate()([u3, c1])

    c5 = Conv2D(filters=n_classes, kernel_size=(3,3), activation='linear', padding='same')(u3_c1)

    model = Model(inputs=x, outputs=c5) 
    #model.summary()
    return model

_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)

def weighted_unet_no_transpose_conv(input_shape, weighted_input_shape, n_classes, base_exp=5):
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


def unet_weighted(input_shape, n_classes):
    inp1 = Input(input_shape)
    weighted_input = Input(shape=(388, 388, 5))
    base = 2
    exp = 6

    # 64 filters
    c1 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='valid')(inp1)
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


def fcnn_functional(n_classes):

    x = Input((None, None, 36))
    base = 2 
    # exp from 4 to 5.
    exp = 5
    c1 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(x)
    c1 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(c1)
    mp1 = MaxPooling2D(pool_size=2, strides=(2, 2))(c1)
    
    exp+=1

    c2 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(mp1)
    c2 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(c2)
    mp2 = MaxPooling2D(pool_size=2, strides=(2, 2))(c2)
    #mp2 = Dropout(0.5)(mp2)

    exp+=1

    c3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(mp2)
    c3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(c3)
    mp3 = MaxPooling2D(pool_size=2, strides=(2, 2))(c3)
    #mp3 = Dropout(0.5)(mp3)

    exp+=1

    c4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(mp3)
    c4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(c4)
    mp4 = MaxPooling2D(pool_size=2, strides=(2, 2))(c4)

    exp+=1

    c5 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(mp4)
    c5 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(c5)
    mp4 = MaxPooling2D(pool_size=2, strides=(2, 2))(c5)

    last_conv = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(mp4)

    u1 = UpSampling2D(size=(2, 2))(last_conv)
    u1 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u1)
    u1 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u1)

    exp-=1

    u1_c5 = Concatenate()([c5, u1])

    u2 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u1_c5)
    u2 = UpSampling2D(size=(2, 2))(u2)
    u2 = Conv2D(filters=base**exp, kernel_size=(3, 3), activation='relu', padding='same')(u2)
    u2 = Conv2D(filters=base**exp, kernel_size=(3, 3), activation='relu', padding='same')(u2)
    u2 = Dropout(0.5)(u2)

    u2_c4 = Concatenate()([u2, c4])

    exp-=1

    u3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u2_c4)
    u3 = UpSampling2D(size=(2, 2))(u3)
    u3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u3)
    u3 = Dropout(0.5)(u3)

    u3_c3 = Concatenate()([u3, c3])
    
    exp-=1

    u4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u3_c3)
    u4 = UpSampling2D(size=(2, 2))(u4)
    u4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u4)
    #u4 = BatchNormalization(axis=3)(u4)

    u4_c2 = Concatenate()([u4, c2])

    exp-=1

    u5 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u4_c2)
    u5 = UpSampling2D(size=(2, 2))(u5)
    u5 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u5)
    u5 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u5)
    #u5 = BatchNormalization(axis=3)(u5)

    u5_c1 = Concatenate()([u5, c1])

    u6 = Conv2D(filters=n_classes, kernel_size=(3, 3), activation='softmax', padding='same')(u5_c1)

    model = Model(inputs=x, outputs=u6) 
    # model.summary()
    return model
