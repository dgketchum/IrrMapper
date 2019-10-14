import os
import argparse
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
    custom_objects = {'mb':masked_binary_xent(pos_weight=1.0), 'binary_acc':binary_acc}
    try:
        model = load_model(args.model, custom_objects=custom_objects)
    except ValueError as e:
        print(e.args)
        raise

    batch_size = 1
    files = glob(os.path.join(args.test_data_path, '*.pkl'))
    test_generator = BinaryDataSequence(batch_size, files,  training=False)
    cmat, prec, recall = confusion_matrix_from_generator(test_generator, batch_size, model,
            n_classes=2)
    print('model {} has p:{}, r:{}'.format(args.model, prec, recall))
