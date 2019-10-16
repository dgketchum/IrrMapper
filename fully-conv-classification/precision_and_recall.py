import os
import argparse
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from tensorflow.keras.models import load_model
from glob import glob


from losses import binary_focal_loss, binary_acc, masked_binary_xent
from data_generators import RandomMajorityUndersamplingSequence, BinaryDataSequence
from train_utils import confusion_matrix_from_generator


if __name__ ==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test-data-path', type=str, required=True)
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()
    if not args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #custom_objects = {'mb':masked_binary_xent(pos_weight=1.0), 'binary_acc':binary_acc}
    custom_objects = {'mb':masked_binary_xent(), 'bfl':binary_focal_loss(), 'binary_acc':binary_acc}
    try:
        model = load_model(args.model, custom_objects=custom_objects)
    except ValueError as e:
        print(e.args)
        raise

    batch_size = 12
    dirs = os.listdir(args.test_data_path)
    is_top_dir = True
    for d in dirs:
        if not os.path.isdir(os.path.join(args.test_data_path, d)):
            is_top_dir = False
            break
    files = []
    if is_top_dir:
        for d in dirs:
            fs = glob(os.path.join(args.test_data_path, d, '*.pkl'))
            files.extend(fs)
    else:
        files = glob(os.path.join(args.test_data_path, '*.pkl'))
    test_generator = BinaryDataSequence(batch_size, majority_file_list=files, minority_file_list=[],  training=False)
    cmat, prec, recall = confusion_matrix_from_generator(test_generator, batch_size, model,
            n_classes=2)
    print('model {} has \n p:{}\n r:{}'.format(args.model, prec, recall))
