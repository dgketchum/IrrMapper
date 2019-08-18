import argparse
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import random
from functools import partial
from tensorflow.keras.callbacks import (TensorBoard, ModelCheckpoint, LearningRateScheduler)
from tensorflow.keras.models import load_model
from data_generators import SatDataSequence
from models import unet_same_padding


_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)


def weighted_loss(target, output):
    out = -tf.reduce_sum(target*output, len(output.get_shape())-1)
    return out


def acc(y_true, y_pred):
    y_pred_sum = tf.reduce_sum(y_pred, axis=-1)
    mask = tf.not_equal(y_pred_sum, 0)
    y_arg = tf.argmax(y_pred, axis=-1)
    y_t_arg = tf.argmax(y_true, axis=-1)
    y_arg_mask = tf.boolean_mask(y_arg, mask)
    y_t_arg_mask = tf.boolean_mask(y_t_arg, mask)
    return K.mean(K.equal(y_t_arg_mask, y_arg_mask))


def lr_schedule(epoch, initial_learning_rate):
    lr = initial_learning_rate 
    if epoch > 9:
        lr /= 32.
    elif epoch > 7:
        lr /= 16.
    elif epoch > 5:
        lr /= 8.
    elif epoch > 3:
        lr /= 4.
    elif epoch > 1:
        lr /= 3.
    print('Learning rate: ', lr)
    return lr


def _list_to_dict(ls):
    dct = {}
    for i, e in enumerate(ls):
        dct[i] = e
    return dct


def _save_model_info(root_directory, accuracy, loss, weights, augment, learning_rate, balance,
        train_mc):
    directory_name = os.path.join("./models", "{:.3f}acc".format(accuracy))
    if os.path.isdir(directory_name):
        directory_name = os.path.join("./models", "{:.5f}acc".format(accuracy))
    filename = os.path.join(directory_name, "run_info_{:.3f}acc.txt".format(accuracy))
    os.rename(root_directory, directory_name)
    print(filename)
    with open(filename, 'w') as f:
        print("acc: {:.3f}".format(accuracy), file=f)
        print("loss: {}".format(loss), file=f)
        print("weights: {}".format(weights), file=f)
        print("augment scheme: {}".format(augment), file=f)
        print("lr: {}".format(learning_rate), file=f)
        print("balance: {}".format(balance), file=f)
        print('train w multiple classes per tile: {}'.format(train_mc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-f', '--file')
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-b", "--balance", type=int)
    parser.add_argument("-w", "--weights", nargs=1)
    parser.add_argument("-a", "--augment", nargs=1)
    args = parser.parse_args()
    learning_rate_scheduler = partial(lr_schedule, initial_learning_rate=args.learning_rate)
    print("--------------------------------------------------")
    args.augment = [bool(int(x)) for x in args.augment[0].split(' ')]
    args.weights = [float(x) for x in args.weights[0].split(' ')]
    print("---------------HYPERPARAMETERS--------------------")
    print(args.learning_rate, args.balance, args.augment, args.weights)
    print("--------------------------------------------------")
    n_classes = 6
    input_shape = (None, None, 51)
    weight_shape = (None, None, n_classes)
    unique_path = str(time.time())
    root_directory = './models/{}/'.format(unique_path)
    model_path = './models/{}/model.h5'.format(unique_path)
    tb_path = './models/{}/graphs/'.format(unique_path)
    if not os.path.isdir(tb_path):
        os.makedirs(tb_path)
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
    tensorboard = TensorBoard(log_dir=tb_path, update_freq=30, profile_batch=0)
    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
    model = unet_same_padding(input_shape, weight_shape, n_classes=n_classes, initial_exp=5)
    # model.summary()
    opt = tf.keras.optimizers.Adam()
    model.compile(opt, loss=weighted_loss, metrics=[acc])
    class_weights = _list_to_dict(args.weights)
    classes_to_augment = _list_to_dict(args.augment)
    batch_size = 3
    balance = args.balance
    train_mc = True
    generator = SatDataSequence('/home/thomas/ssd/training_data/train_mc/', batch_size=batch_size,
            class_weights=class_weights, balance=balance, n_classes=n_classes,
            classes_to_augment=classes_to_augment)
    class_weights = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1}
    valid_generator = SatDataSequence('/home/thomas/ssd/training_data/test_mc/', 
            batch_size=batch_size, balance=False, n_classes=n_classes,
            single_class_per_tile=True, class_weights=class_weights)
    model.fit_generator(generator,
            epochs=7,
            callbacks=[lr_scheduler, checkpoint, tensorboard],
            use_multiprocessing=True,
            validation_data=valid_generator,
            workers=12,
            shuffle=False,
            verbose=1)
    model = load_model(model_path, custom_objects={'weighted_loss':weighted_loss, 'tf':tf,
        '_epsilon':_epsilon, 'acc':acc})
    accuracy = model.evaluate_generator(valid_generator)
    loss = accuracy[0]
    accuracy = accuracy[1]
    _save_model_info(root_directory, accuracy, loss, class_weights, classes_to_augment,
            args.learning_rate, args.balance, train_mc)
