import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Input, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout, UpSampling2D)

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
        activation='relu')) # 1x1 convolutions for pixel-wise prediciton.
    # Take a look at the model summary
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
    mp2 = Dropout(0.5)(mp2)

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
    u2 = Dropout(0.5)(u2)

    u2_c2 = Concatenate()([u2, c2])
    u2_c2 = Dropout(0.5)(u2_c2)

    c4 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(u2_c2)
    u3 = UpSampling2D(size=(2, 2))(c4)
    u3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(u3)

    u3_c1 = Concatenate()([u3, c1])

    c5 = Conv2D(filters=n_classes, kernel_size=(3,3), activation='relu', padding='same')(u3_c1)

    model = Model(inputs=x, outputs=c5) 
    #model.summary()
    return model


def fcnn_functional(n_classes):

    x = Input((None, None, 36))
    base = 2 
    # exp from 4 to 5.
    exp = 6
    c1 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(x)
    c1 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(c1)
    mp1 = MaxPooling2D(pool_size=2, strides=(2, 2))(c1)
    
    exp+=1

    c2 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(mp1)
    c2 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(c2)
    mp2 = MaxPooling2D(pool_size=2, strides=(2, 2))(c2)
   # mp2 = Dropout(0.5)(mp2)

    exp+=1

    c3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(mp2)
    c3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(c3)
    mp3 = MaxPooling2D(pool_size=2, strides=(2, 2))(c3)
   #Jkj mp3 = Dropout(0.5)(mp3)

    exp+=1

    c4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(mp3)
    c4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(c4)
    mp4 = MaxPooling2D(pool_size=2, strides=(2, 2))(c4)
   # mp4 = Dropout(0.5)(mp4)

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
   # u2 = Dropout(0.5)(u2)

    u2_c4 = Concatenate()([u2, c4])

    exp-=1

    u3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u2_c4)
    u3 = UpSampling2D(size=(2, 2))(u3)
    u3 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u3)
   #u3 = Dropout(0.5)(u3)

    u3_c3 = Concatenate()([u3, c3])
    
    exp-=1

    u4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u3_c3)
    u4 = UpSampling2D(size=(2, 2))(u4)
    u4 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u4)

    u4_c2 = Concatenate()([u4, c2])

    exp-=1

    u5 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u4_c2)
    u5 = UpSampling2D(size=(2, 2))(u5)
    u5 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u5)
    u5 = Conv2D(filters=base**exp, kernel_size=(3,3), activation='relu', padding='same')(u5)

    u5_c1 = Concatenate()([u5, c1])

    u6 = Conv2D(filters=n_classes, kernel_size=(3,3), activation='relu', padding='same')(u5_c1)

    model = Model(inputs=x, outputs=u6) 
    #model.summary()
    return model
