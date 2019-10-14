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
from data_generators import RandomMajorityUndersamplingSequence, BinaryDataSequence
from train_utils import lr_schedule
from losses import (binary_focal_loss, binary_acc, masked_binary_xent, masked_categorical_xent,
        multiclass_acc)

# don't monitor binary acc any more, monitor precision and recall.

if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('--training-dir', type=str)
    ap.add_argument('--testing-dir', type=str)
    args = ap.parse_args()

    training_dir = args.training_dir
    testing_dir = args.testing_dir

    initial_learning_rate = 1e-3

    input_shape = (None, None, 51)

    n_classes = 1

    model = unet(input_shape, initial_exp=4, n_classes=n_classes)
    model_path = 'random_majority_files/only_irrigated_no_border_labels/'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    
    model_path += 'model.h5'

    tensorboard = TensorBoard(log_dir='/tmp/', 
            profile_batch=0,
            update_freq=30,
            batch_size=3)
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_binary_acc',
                                 verbose=1,
                                 save_best_only=True)
    lr_schedule = partial(lr_schedule, initial_learning_rate=initial_learning_rate)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    train_files = glob('/home/thomas/ssd/binary_train_no_border_labels/train/class_1_data/*.pkl')
    test_files = glob('/home/thomas/ssd/binary_train_no_border_labels/test/class_1_data/*.pkl')

    opt = tf.keras.optimizers.Adam()
    batch_size = 4
    loss_func = masked_binary_xent(pos_weight=1.0)
    # loss_func = masked_categorical_xent
    model.compile(opt, loss=loss_func, metrics=[binary_acc])
    # train_generator = RandomMajorityUndersamplingSequence(batch_size, training_dir)
    # test_generator = RandomMajorityUndersamplingSequence(batch_size, testing_dir)
    train_generator = BinaryDataSequence(batch_size, train_files)
    test_generator = BinaryDataSequence(batch_size, test_files)
    model.fit_generator(train_generator, 
            epochs=50,
            validation_data=test_generator,
            callbacks=[tensorboard, lr_scheduler, checkpoint],
            use_multiprocessing=True,
            workers=12,
            max_queue_size=30,
            verbose=1)
