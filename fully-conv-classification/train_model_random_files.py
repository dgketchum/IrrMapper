import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from tensorflow.keras.callbacks import (TensorBoard, ModelCheckpoint, LearningRateScheduler)
from functools import partial
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from scipy.special import expit
from random import sample
from glob import glob


from models import unet
from data_generators import DataGenerator
from train_utils import lr_schedule
from losses import (binary_focal_loss, binary_acc, masked_binary_xent, masked_categorical_xent,
        multiclass_acc)

join = os.path.join
# don't monitor binary acc any more, monitor precision and recall.
# or monitor top-k accuracy.

if __name__ == '__main__':

    initial_learning_rate = 1e-3

    input_shape = (None, None, 51)

    n_classes = 4

    model = unet(input_shape, initial_exp=4, n_classes=n_classes)
    model_path = 'random_majority_files/multiclass/'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    model_path += 'three_scenes_concat_only_irrigated_tiles_balanced_pixels_per_batch.h5'

    tensorboard = TensorBoard(log_dir='/tmp/', 
            profile_batch=0,
            update_freq=30,
            batch_size=3)
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_multiclass_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_schedule = partial(lr_schedule, initial_learning_rate=initial_learning_rate)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=True)

    root = '/home/thomas/ssd/multiclass_no_border_labels/'
    train_dir = join(root, 'train')
    test_dir = join(root, 'test')

    opt = tf.keras.optimizers.Adam()
    batch_size = 4
    loss_func = masked_categorical_xent
    metric = multiclass_acc
    model.compile(opt, loss=loss_func, metrics=[metric])
    train_generator = DataGenerator(train_dir, batch_size, target_classes=0, n_classes=n_classes,
            balance_examples_per_batch=False, balance_pixels_per_batch=True)
    test_generator = DataGenerator(test_dir, batch_size, target_classes=0, n_classes=n_classes, training=False)
    model.fit_generator(train_generator, 
            epochs=25,
            validation_data=test_generator,
            callbacks=[tensorboard, lr_scheduler, checkpoint],
            use_multiprocessing=True,
            workers=12,
            max_queue_size=12,
            verbose=1)
