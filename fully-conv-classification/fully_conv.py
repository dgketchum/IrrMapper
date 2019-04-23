import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (TensorBoard, ModelCheckpoint, LearningRateScheduler)
from data_generators import generate_unbalanced_data, SatDataSequence
from models import unet_same_padding


def weighted_loss(target, output):
    out = -tf.reduce_sum(target*output, len(output.get_shape())-1)
    return out


def weighted_focal_loss(target, output, gamma=1):
    # L = a0 *(1-pt)^gamma * ce
    # Output of model is CE.
    # Target is one-hot encoded.
    soft = tf.nn.softmax(output, axis=-1)
    pt = tf.pow(1-soft, gamma) # probability
    return -tf.reduce_sum(target*output*pt, len(output.get_shape())-1)
    

def acc(y_true, y_pred):
    y_pred_sum = tf.reduce_sum(y_pred, axis=-1)
    mask = tf.not_equal(y_pred_sum, 0)
    y_arg = tf.argmax(y_pred, axis=-1)
    y_t_arg = tf.argmax(y_true, axis=-1)
    y_arg_mask = tf.boolean_mask(y_arg, mask)
    y_t_arg_mask = tf.boolean_mask(y_t_arg, mask)
    return K.mean(K.equal(y_t_arg_mask, y_arg_mask))


def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 100:
        lr /= 64
    if epoch > 45:
        lr /= 32.
    elif epoch > 30:
        lr /= 16.
    elif epoch > 15:
        lr /= 8.
    elif epoch > 10:
        lr /= 4.
    elif epoch > 5:
        lr /= 2.
    print('Learning rate: ', lr)
    return lr

if __name__ == '__main__':

    n_classes = 4
    input_shape = (None, None, 51)
    weight_shape = (None, None, n_classes)
    filepath = './models/normal_loss_20_irr_weight.h5'
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
    tensorboard = TensorBoard(log_dir='graphs/{}'.format(time.time()))
    lr_scheduler = LearningRateScheduler(lr_schedule)
    model = unet_same_padding(input_shape, weight_shape, n_classes=n_classes, initial_exp=5)
    #model.summary()
    opt = tf.keras.optimizers.Adam(1e-4)
    model.compile(opt, loss=weighted_loss, metrics=[acc])
    class_weights = {0:20, 1:1.0, 2:2.96, 3:50} 
    class_weights_valid = {0:1.0, 1:1.0, 2:1.0, 3:1.0} 
    batch_size = 4
    generator = SatDataSequence('training_data/train/', batch_size=batch_size,
            class_weights=class_weights)
    valid_generator = SatDataSequence('training_data/test/', batch_size=batch_size,
            class_weights=class_weights)
    model.fit_generator(generator,
            epochs=50,
            validation_data=valid_generator,
            callbacks=[checkpoint, lr_scheduler, tensorboard],
            verbose=1)
