import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
import numpy as np
from argparse import ArgumentParser
from tensorflow.keras.callbacks import (TensorBoard, ModelCheckpoint, LearningRateScheduler)
from functools import partial
from tensorflow.keras.models import load_model
from glob import glob


from models import unet
from data_generators import RandomMajorityUndersamplingSequence, BinaryDataSequence
from train_utils import lr_schedule
from losses import (binary_focal_loss, binary_acc, masked_binary_xent, masked_categorical_xent,
        multiclass_acc)

def make_file_list(directory, ext='*.pkl'):

    is_top_dir = True
    for d in os.listdir(directory):
        if not os.path.isdir(os.path.join(directory, d)):
            is_top_dir = False
            break
    
    files = []
    if is_top_dir:
        for d in os.listdir(directory):
            files.extend(glob(os.path.join(directory, d, ext)))
    else:
        files.extend(glob(os.path.join(directory, ext)))
    return files



if __name__ == '__main__':
    
    ap = ArgumentParser()
    ap.add_argument("--model-to-finetune", type=str, required=True)
    ap.add_argument("--loss-func", type=str, required=True)
    args = ap.parse_args()

    mb = masked_binary_xent(1.0)
    bfl = binary_focal_loss()
    input_shape = (None, None, 51)

    n_classes = 1
    custom_objects = {'mb':masked_binary_xent(), 'bfl':bfl, 'binary_acc':binary_acc}
    model_frozen = load_model(args.model_to_finetune, custom_objects=custom_objects)

    if 'finetuned' not in args.model_to_finetune:
        model_out_path = os.path.splitext(args.model_to_finetune)[0] + '_finetuned.h5'
    else:
        model_out_path = args.model_to_finetune

    print(model_out_path)

    for layer in model_frozen.layers[:-1]:
        layer.trainable = False

    checkpoint = ModelCheckpoint(filepath=model_out_path,
                                 monitor='val_binary_acc',
                                 verbose=1,
                                 save_best_only=True)

    initial_learning_rate = 1e-3
    lr_schedule = partial(lr_schedule, initial_learning_rate=initial_learning_rate)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    batch_size = 4
    data_directory = '/home/thomas/ssd/binary_train_no_border_labels/'
    minority_file_list = glob(os.path.join(data_directory, 'train/class_1_data/*pkl') )
    majority_file_list = glob(os.path.join(data_directory, 'train/class_0_data/*pkl') )
    train_generator = BinaryDataSequence(batch_size, minority_file_list, majority_file_list, balance_files=True, erode=False)
    minority_file_list = glob(os.path.join(data_directory, 'test/class_1_data/*pkl') )
    majority_file_list = glob(os.path.join(data_directory, 'test/class_0_data/*pkl') )
    opt = tf.keras.optimizers.Adam()
    loss_func = binary_focal_loss(gamma=3, alpha=0.99)
    # loss_func = masked_categorical_xent
    model_frozen.compile(opt, loss=loss_func, metrics=[binary_acc])
    # train_generator = RandomMajorityUndersamplingSequence(batch_size, training_dir)
    # test_generator = RandomMajorityUndersamplingSequence(batch_size, testing_dir)
    test_generator = BinaryDataSequence(batch_size, minority_file_list, majority_file_list,
            training=False, erode=False, total_files=120)
    model_frozen.fit_generator(train_generator, 
            epochs=15,
            validation_data=test_generator,
            callbacks=[lr_scheduler, checkpoint],
            use_multiprocessing=True,
            workers=12,
            max_queue_size=30,
            verbose=1)
    model_frozen.save("fuullytaruined.h5")
