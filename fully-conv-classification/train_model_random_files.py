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
from losses import *

join = os.path.join
# don't monitor binary acc any more, monitor precision and recall.
# or monitor top-k accuracy.

if __name__ == '__main__':

    initial_learning_rate = 1e-3

    input_shape = (None, None, 51)

    n_classes = 4

    ap = ArgumentParser()
    ap.add_argument('--gamma', type=float)

    args = ap.parse_args()

    gamma = float(args.gamma)
    if gamma is None:
        # default gamma
        gamma = 2.0

    model = unet(input_shape, initial_exp=4, n_classes=n_classes)
    model_path = 'random_majority_files/multiclass/'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    model_path += 'three_scenes_concat_balance_examples_per_batch_focal_loss_gamma-{}.h5'.format(gamma)

    tensorboard = TensorBoard(log_dir='/tmp/', 
            profile_batch=0,
            update_freq=30,
            batch_size=3)
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_multiclass_acc',
                                 verbose=1,
                                 save_best_only=True)

    epochs = 1000
    lr_schedule = partial(lr_schedule, initial_learning_rate=initial_learning_rate, efold=epochs/10)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=True)

    root = '/home/thomas/ssd/multiclass_no_border_labels/'
    train_dir = join(root, 'train')
    test_dir = join(root, 'test')

    opt = tf.keras.optimizers.Adam()
    batch_size = 4
    loss_func = multiclass_focal_loss(gamma=gamma)
    metric = multiclass_acc
    model.compile(opt, loss=loss_func, metrics=[metric])
    train_generator = DataGenerator(train_dir, batch_size, target_classes=None, 
            n_classes=n_classes, balance=False, balance_pixels_per_batch=False, 
            balance_examples_per_batch=True, apply_irrigated_weights=False,
            training=False)
    test_generator = DataGenerator(test_dir, batch_size, target_classes=0, 
            n_classes=n_classes, training=False)
    model.fit_generator(train_generator, 
            epochs=epochs,
            validation_data=test_generator,
            callbacks=[tensorboard, lr_scheduler, checkpoint],
            use_multiprocessing=False,
            workers=1,
            max_queue_size=1,
            verbose=0)
