import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
from time import time


from models import unet, two_headed_unet
from data_generators import DataGenerator
from train_utils import lr_schedule, F1Score
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

    model = two_headed_unet(input_shape, initial_exp=4, n_classes=n_classes)
    model_path = 'random_majority_files/multiclass/normal_xen_with_cdl/'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    model_path += 'balance_examples_per_batch_xen.h5'

    pth = '/home/thomas/tensorboard/'+str(time())
    if not os.path.isdir(pth):
        os.mkdir(pth)
    tensorboard = TensorBoard(log_dir=pth,
            profile_batch=0,
            update_freq=30,
            batch_size=3)
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_irr_m_acc',
                                 verbose=1,
                                 save_best_only=True)

    epochs = 1000
    lr_schedule = partial(lr_schedule, initial_learning_rate=initial_learning_rate, efold=epochs/10)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=True)

    root = '/home/thomas/share/multiclass_with_separate_fallow_directory_and_cdl/'
    train_dir = join(root, 'train')
    test_dir = join(root, 'test')

    opt = tf.keras.optimizers.Adam()
    batch_size = 4
    loss_func = masked_categorical_xent
    metric = m_acc
    loss_weights = [1.0, 0.25]
    model.compile(opt, loss=[masked_categorical_xent, 'binary_crossentropy'],
            metrics={'irr':metric, 'cdl':'accuracy'}, loss_weights=loss_weights)
    train_generator = DataGenerator(train_dir, batch_size, target_classes=None, 
            n_classes=n_classes, balance=False, balance_pixels_per_batch=False, 
            balance_examples_per_batch=True, apply_irrigated_weights=False,
            training=True, augment_data=False, use_cdl=True)
    test_generator = DataGenerator(test_dir, batch_size, target_classes=None, 
            n_classes=n_classes, training=False, balance=False, steps_per_epoch=30,
            augment_data=False, use_cdl=True)
    m2 = F1Score(test_generator, n_classes, model_path, batch_size, two_headed_net=True)
    model.fit_generator(train_generator, 
            epochs=epochs,
            validation_data=test_generator,
            callbacks=[tensorboard, lr_scheduler, checkpoint, m2],
            use_multiprocessing=False,
            workers=1,
            max_queue_size=1,
            verbose=1)
