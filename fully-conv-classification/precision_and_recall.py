import os
import argparse
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from tensorflow.keras.models import load_model
from glob import glob


from losses import (binary_focal_loss, binary_acc, masked_binary_xent, multiclass_acc,
        masked_categorical_xent)
from data_generators import DataGenerator
from train_utils import confusion_matrix_from_generator


if __name__ ==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test-data-path', type=str, required=True)
    parser.add_argument('--target-class', type=str)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--n-classes', type=int, default=4)
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()
    if not args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #custom_objects = {'mb':masked_binary_xent(pos_weight=1.0), 'binary_acc':binary_acc}
    custom_objects = {'mb':masked_binary_xent, 'multiclass_acc':multiclass_acc,
            'binary_acc':binary_acc, 'masked_categorical_xent':masked_categorical_xent}
    try:
        model = load_model(args.model, custom_objects=custom_objects)
    except ValueError as e:
        print(e.args)
        raise
    batch_size = args.batch_size
    test_generator = DataGenerator(data_directory=args.test_data_path, batch_size=batch_size,
            training=False, target_classes=args.target_class)
    cmat, prec, recall = confusion_matrix_from_generator(test_generator, batch_size, model,
            n_classes=args.n_classes)
    print('model {} has \n p:{}\n r:{}'.format(args.model, prec, recall))
