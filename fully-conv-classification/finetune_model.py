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
from data_generators import DataGenerator
from train_utils import lr_schedule
from losses import (binary_focal_loss, binary_acc, masked_binary_xent, masked_categorical_xent,
        multiclass_acc)

join = os.path.join
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
    ap.add_argument("--loss-func", type=str)
    args = ap.parse_args()

    input_shape = (None, None, 51)

    n_classes = 1
    custom_objects = {'mb':masked_binary_xent(), 'binary_acc':binary_acc,
            'masked_categorical_xent':masked_categorical_xent, 'multiclass_acc':multiclass_acc}

    model_frozen = load_model(args.model_to_finetune, custom_objects=custom_objects)

    if 'finetuned' not in args.model_to_finetune:
        model_out_path = os.path.splitext(args.model_to_finetune)[0] + '_finetuned'
    else:
        model_out_path = args.model_to_finetune

    model_out_path = os.path.join("random_majority_files/SGD/", 'finetuned')
    print(model_out_path)

    for layer in model_frozen.layers[:-2]: 
        layer.trainable = False

    # model_frozen.summary()

    model_out_path += "-{epoch:02d}-{val_multiclass_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath=model_out_path,
                                 monitor='val_multiclass_acc',
                                 verbose=1, save_best_only=False)

    initial_learning_rate = 1e-3
    lr_schedule = partial(lr_schedule, initial_learning_rate=initial_learning_rate, efold=50)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=True)
    opt = tf.keras.optimizers.SGD()

    root = '/home/thomas/ssd/multiclass_no_border_labels/'
    train_dir = join(root, 'train')
    test_dir = join(root, 'test')

    loss_func = masked_categorical_xent
    batch_size = 8
    model_frozen.compile(opt, loss=loss_func, metrics=[multiclass_acc])
    train_generator = DataGenerator(train_dir, batch_size, target_classes=None, 
            n_classes=n_classes, training=True, apply_irrigated_weights=True,
            steps_per_epoch=200)
    test_generator = DataGenerator(test_dir, batch_size, target_classes=[0, 1],
            n_classes=n_classes, training=False)
    model_frozen.fit_generator(train_generator, 
            epochs=40,
            validation_data=test_generator,
            callbacks=[lr_scheduler, checkpoint],
            use_multiprocessing=False,
            workers=1,
            max_queue_size=30,
            verbose=1)
