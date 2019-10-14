import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import keras.backend as K
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.callbacks import (TensorBoard, ModelCheckpoint, LearningRateScheduler)
from functools import partial
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from scipy.special import expit

from models import unet
from data_generators import SatDataSequence
from data_utils import map_bands_to_indices
from train_utils import (construct_parser, make_temporary_directory, save_model_info,
        lr_schedule, confusion_matrix_from_generator)
from runspec import target_bands
from losses import *



training_data_dir = '/home/thomas/ssd/single_scene/train/'
test_data_dir = '/home/thomas/ssd/single_scene/test/'


if __name__ == '__main__':

    parser = construct_parser()
    args = parser.parse_args()
    initial_learning_rate = args.learning_rate

    temp_dir, model_path, tensorboard_path = make_temporary_directory('./models/')

    input_shape = (None, None, 19)

    n_classes = 1

    model = unet(input_shape, initial_exp=4, n_classes=n_classes)

    tensorboard = TensorBoard(log_dir=tensorboard_path, 
            profile_batch=0,
            update_freq=30,
            batch_size=3)
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_binary_acc',
                                 verbose=1,
                                 save_best_only=True)
    lr_schedule = partial(lr_schedule, initial_learning_rate=initial_learning_rate)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    opt = tf.keras.optimizers.Adam()
    class_weights = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1} 
    classes_to_augment = True #{0:1, 1:1, 2:1, 3:1, 4:1, 5:1} 
    batch_size = 4
    pos_weight = args.pos_weight
    loss_func = binary_focal_loss(gamma=2, alpha=0.25)
    model.compile(opt, loss=loss_func, metrics=[binary_acc])
    train_generator = SatDataSequence(training_data_dir, class_weights=class_weights,
            batch_size=batch_size, n_classes=n_classes)
    test_generator = SatDataSequence(test_data_dir, class_weights=class_weights,
            batch_size=batch_size, training=False, n_classes=n_classes)

    model.fit_generator(train_generator, 
            epochs=5,
            validation_data=test_generator,
            callbacks=[tensorboard, lr_scheduler, checkpoint],
            use_multiprocessing=True,
            workers=12,
            max_queue_size=30,
            verbose=1)
    model.save('single_scene.h5')
    # loss_func = binary_focal_loss(gamma=2, alpha=0.25)
    # custom_objects = {'bfl':loss_func, 'binary_acc':binary_acc}
    # model = load_model('models/temp/model.h5', custom_objects=custom_objects)
    # test_data_dir = '/home/thomas/ssd/binary_train/test/'
    # test_generator = SatDataSequence(test_data_dir, class_weights=class_weights,
    #         batch_size=batch_size, training=False, n_classes=n_classes)
    # accuracy = model.evaluate_generator(test_generator)
    # loss = accuracy[0]
    # accuracy = accuracy[1]
    # test_generator = SatDataSequence(test_data_dir, class_weights=class_weights,
    #         batch_size=1, n_classes=n_classes)
    # cmat, prec, recall = confusion_matrix_from_generator(test_generator, model, n_classes=2)
    # print(prec, recall)
    # save_model_info(temp_dir, loss_func.__name__, accuracy, loss, class_weights,
    #         classes_to_augment, pos_weight, initial_learning_rate, cmat, prec, recall)
