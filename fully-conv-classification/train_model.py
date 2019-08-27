import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import keras.backend as K
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.callbacks import (TensorBoard, ModelCheckpoint, LearningRateScheduler)
from functools import partial
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from models import unet
from data_generators import SatDataSequence
config = tf.ConfigProto()
config.gpu_options.allow_growth

_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)

def _softmax(logits):
    preds = np.exp(logits)
    soft = preds / np.sum(preds, axis=-1, keepdims=True)
    return soft


def _confusion_matrix(valid_generator, model):
    first = True
    for batch_x, y_true in valid_generator:
        y_true = y_true[0][0]
        mask = np.sum(y_true, axis=2).astype(bool)
        y_pred = model.predict(batch_x)
        y_pred = y_pred[0]
        y_pred = _softmax(y_pred)
        y_pred = np.argmax(y_pred, axis=2)
        y_true = np.argmax(y_true, axis=2)
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        if first:
            cmat = confusion_matrix(y_true, y_pred,
                    labels=[0, 1, 2, 3, 4, 5])
            first = False
        else:
            cmat += confusion_matrix(y_true, y_pred,
                    labels=[0, 1, 2, 3, 4, 5])
    print(cmat)
    precision_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    recall_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for i in range(6):
        recall_dict[i] = cmat[i, i] / np.sum(cmat[i, :])
        precision_dict[i] = cmat[i, i] / np.sum(cmat[:, i])
    return cmat, recall_dict, precision_dict


def weighted_loss(target, output):
    out = -tf.reduce_sum(target*output, len(output.get_shape())-1)
    return out


def weighted_loss_ce_and_dl(target, output):
    # Target: One hot encoding of segmentation mask.
    # Output: Output of network. In this case, log(softmax).
    soft = tf.nn.softmax(output)
    numerator = tf.reduce_sum(soft*target, 1)
    numerator = tf.reduce_sum(numerator, 2)
    sum_ui_k = tf.reduce_sum(soft, 1)
    sum_ui_k = tf.reduce_sum(sum_ui_k, 2)
    sum_vi_k = tf.reduce_sum(target, 1)
    sum_vi_k = tf.reduce_sum(sum_vi_k, 2)

    final = (-2/4)*tf.reduce_sum(numerator / (sum_ui_k + sum_vi_k), 1)
    out = -tf.reduce_sum(target*output, len(output.get_shape())-1)
    return final + out


def weighted_focal_loss(target, output, gamma=1):
    # L = a0 *(1-pt)^gamma * ce
    # Output of model is CE.
    # Target is one-hot encoded.
    soft = tf.nn.softmax(output, axis=-1)
    pt = tf.pow(1-soft, 2) # probability
    return -tf.reduce_sum(target*output*pt, len(output.get_shape())-1)
    

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
    if epoch > 15:
        lr /= 256
    elif epoch > 13:
        lr /= 128
    elif epoch > 11:
        lr /= 64
    elif epoch > 9:
        lr /= 32.
    elif epoch > 7:
        lr /= 16.
    elif epoch > 5:
        lr /= 8.
    elif epoch > 3:
        lr /= 4.
    elif epoch > 1:
        lr /= 2.
    print('Learning rate: ', lr)
    return lr


def _save_model_info(root_directory, train_type, loss_func, accuracy, loss, class_weights, classes_to_augment, initial_learning_rate, train_mc, cmat, precision, recall):
    directory_name = os.path.join("./models", "{:.3f}acc_class_weights_multiple_classes".format(accuracy))
    if os.path.isdir(directory_name):
        directory_name = os.path.join("./models", "{:.5f}acc".format(accuracy))
    filename = os.path.join(directory_name, "run_info_{:.3f}acc.txt".format(accuracy))
    os.rename(root_directory, directory_name)
    print(filename)
    with open(filename, 'w') as f:
        print("acc: {:.3f}".format(accuracy), file=f)
        print("train_type: {}".format(train_type), file=f)
        print("loss_func: {}".format(loss_func), file=f)
        print("loss: {}".format(loss), file=f)
        print("weights: {}".format(class_weights), file=f)
        print("augment scheme: {}".format(classes_to_augment), file=f)
        print("lr: {}".format(initial_learning_rate), file=f)
        print('train w multiple classes per tile: {}'.format(train_mc), file=f)
        print('confusion_matrix: {}'.format(cmat), file=f)
        print('precision: {}'.format(precision), file=f)
        print('recall: {}'.format(recall), file=f)


def _return_loss(inp):
    if 'focal' in inp:
        return weighted_focal_loss
    return weighted_loss


def _list_to_dict(ls):
    dct = {}
    for i, e in enumerate(ls[0].split(' ')):
        dct[i] = e
    return dct
    

def _nclass(i):
    if 'multiple' in i:
        return 6
    return 5



if __name__ == '__main__':

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-f', '--file')
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-b", "--balance", type=int)
    parser.add_argument("-w", "--weights", nargs=1)
    parser.add_argument("-lf", "--loss_func")
    parser.add_argument("-tt", '--train_type')
    parser.add_argument("-nc", '--n_classes')
    parser.add_argument("-td", '--train-data-dir', default='/home/thomas/ssd/training_data/train/')
    parser.add_argument("-ted", '-test-data-dir', default='/home/thomas/ssd/training_data/test/')
    args = parser.parse_args()
    learning_rate_scheduler = partial(lr_schedule, initial_learning_rate=args.learning_rate)
    # runtime params
    n_classes = _nclass(args.train_type)
    input_shape = (None, None, 51)
    initial_learning_rate = args.learning_rate
    class_weights = {0:85, 1:1.0, 2:1.546, 3:784.286, 4:972.84, 5:357.78} 
    class_weights = _list_to_dict(args.weights)
    classes_to_augment = {0:False, 1:False, 2:False, 3:False, 4:False, 7:True}
    loss_func = _return_loss(args.loss_func)
    print(initial_learning_rate, class_weights, loss_func)
    #
    weight_shape = (None, None, n_classes)
    filepath = './models/template_to_fill_in/model.h5'
    tb_path = './models/template_to_fill_in/' + str(time.time())
    root_directory = './models/template_to_fill_in/'
    if not os.path.isdir(tb_path):
        os.makedirs(tb_path)
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
    tensorboard = TensorBoard(log_dir=tb_path, profile_batch=0, update_freq=30, batch_size=3)
    lr_schedule_func = partial(lr_schedule, initial_learning_rate=initial_learning_rate)
    lr_scheduler = LearningRateScheduler(lr_schedule_func)
    model = unet_same_padding(input_shape, weight_shape, n_classes=n_classes, initial_exp=4)
    opt = tf.keras.optimizers.Adam()
    model.compile(opt, loss=loss_func, metrics=[acc])
    batch_size = 4
    balance = True
    train_mc = True
    training_data_directory = args.train_type
    generator = SatDataSequence('{}'.format(args.train_data_dir), hatch_size=batch_size,
            class_weights=class_weights, single_class_per_tile=False, balance=balance,
            n_classes=n_classes, classes_to_augment=classes_to_augment)
    class_weights = {0:1, 1:1.0, 2:1.0, 3:1, 4:1, 5:1} 
    valid_generator = SatDataSequence('{}'.format(args.test_data_dir), 
            batch_size=batch_size, balance=False, n_classes=n_classes, single_class_per_tile=False,
            class_weights=class_weights)

    model.fit_generator(generator,
            epochs=20, 
            callbacks=[lr_scheduler, checkpoint, tensorboard],
            use_multiprocessing=True,
            validation_data=valid_generator,
            workers=12,
            max_queue_size=30,
            verbose=1)

    if args.loss_func == 'focal':
        model = load_model(filepath, custom_objects={'weighted_focal_loss':weighted_focal_loss, 
            'tf':tf, '_epsilon':_epsilon, 'acc':acc})
    else:
        model = load_model(filepath, custom_objects={'weighted_loss':weighted_loss, 
            'tf':tf, '_epsilon':_epsilon, 'acc':acc})

    accuracy = model.evaluate_generator(valid_generator)
    loss = accuracy[0]
    accuracy = accuracy[1]
    cmat, prec, recall = _confusion_matrix(valid_generator, model)
    _save_model_info(root_directory, args.train_type, args.loss_func, accuracy, loss, class_weights,
            classes_to_augment, initial_learning_rate, train_mc, cmat, prec, recall)
