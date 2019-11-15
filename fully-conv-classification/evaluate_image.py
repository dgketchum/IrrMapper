import os
import numpy as np
import keras.backend as K
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
import pdb
import argparse

from sys import stdout
from tensorflow.keras.models import load_model
from glob import glob
from rasterio.errors import RasterioIOError
from matplotlib.pyplot import imshow, show, subplots
from multiprocessing import Pool
from scipy.special import expit
from scipy.stats import mode

from train_utils import softmax
from runspec import irrigated_path_rows_mt
from data_utils import (save_raster, stack_rasters, stack_rasters_multiprocess,
        paths_map_multiple_scenes, load_raster, clip_raster, paths_mapping_single_scene,
        mean_of_three)
from losses import *
from extract_training_data import concatenate_fmasks

_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)

masked_binary_xent = masked_binary_xent(pos_weight=1.0)
custom_objects = {'masked_binary_xent':masked_binary_xent, 'binary_acc':binary_acc}

def _evaluate_image_return_logits(model, raster, n_classes, n_overlaps=4):
    chunk_size = 608
    diff = 608
    stride = 608
    overlap_step = 10
    raster = np.swapaxes(raster, 0, 2)
    raster = np.expand_dims(raster, 0)
    out = np.zeros((raster.shape[1], raster.shape[2], n_classes))
    for k in range(0, n_overlaps*overlap_step, overlap_step):
        for i in range(k, raster.shape[1]-diff, stride):
            for j in range(k, raster.shape[2]-diff, stride):
                sub_raster = raster[:, i:i+chunk_size, j:j+chunk_size, :]
                preds = model.predict([sub_raster]) 
                out[i:i+chunk_size, j:j+chunk_size, :] += preds[0]
            stdout.write("K: {} of {}. Percent done: {:.2f}\r".format(k // overlap_step + 1, n_overlaps, i / raster.shape[1]))
    out = np.swapaxes(out, 0, 2)
    out = out.astype(np.float32)
    return out


def fmask_evaluated_image(evaluated_image, path, row, year, landsat_directory):
    image, meta = load_raster(evaluated_image)
    suffix = str(path) + '_' + str(row) + '_' + str(year)
    image_subdirectory = os.path.join(landsat_directory, suffix)
    temp_mask = np.expand_dims(np.zeros_like(image)[0], 0)
    meta.update(count=1)
    masked_image = concatenate_fmasks(image_subdirectory, temp_mask, meta, nodata=1)
    for i in range(image.shape[0]):
        image[i, :, :][masked_image.mask[0]] = np.nan
    meta.update(count=image.shape[0])
    meta.update(nodata=np.nan)
    return image, meta
    

def evaluate_image_many_shot(image_directory, model_paths, n_classes=4,
        n_overlaps=4, outfile=None, custom_objects=None, preprocessing_func=None):
    ''' To recover from same padding, slide many different patches over the image. '''
    print(outfile)
    if not isinstance(model_paths, list):
        model_paths = [model_paths]
    if os.path.isfile(outfile):
        print("image {} already exists".format(outfile))
        return
    if not os.path.isdir(image_directory):
        print('Images not downloaded for {}'.format(image_directory))
        return
    paths_mapping = paths_map_multiple_scenes(image_directory)
    template, meta = load_raster(paths_mapping['B1.TIF'][0])
    image_stack = stack_rasters_multiprocess(paths_mapping, meta, template.shape)
    if preprocessing_func is not None:
        image_stack = mean_of_three(image_stack, paths_mapping)
    out_arr = np.zeros((n_classes, image_stack.shape[1], image_stack.shape[2]))
    for i, model_path in enumerate(model_paths):
        print('loading {}'.format(model_path))
        model = load_model(model_path, custom_objects=custom_objects)
        out_arr += _evaluate_image_return_logits(model, image_stack, n_classes=n_classes,
            n_overlaps=n_overlaps)
        del model

    out_arr = softmax(out_arr)
    temp_mask = np.zeros((1, out_arr.shape[1], out_arr.shape[2]))
    fmasked_image = concatenate_fmasks(image_directory, temp_mask, meta, nodata=1)
    for i in range(out_arr.shape[0]):
        out_arr[i, :, :][fmasked_image.mask[0]] = np.nan

    out_arr = out_arr.astype(np.float32)
    meta.update(dtype=np.float32)

    out_arr /= n_overlaps
    if outfile:
        save_raster(out_arr, outfile, meta, count=n_classes)
    return out_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-im', '--image-dir', type=str, required=True)
    parser.add_argument('-o', '--out-dir', type=str)
    parser.add_argument('-n', '--n-classes', type=int, default=5)
    parser.add_argument('-b', '--binary', action='store_true')
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--include-path-row', action='store_true')
    parser.add_argument('--evaluate-all-mt', action='store_true')
    parser.add_argument('--preprocessing-func', type=str)
    parser.add_argument('--year', type=int, default=2013)
    args = parser.parse_args()
    if args.out_dir is None:
        out_dir = os.path.dirname(os.path.splitext(args.model)[0])
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
    else: 
        out_dir = args.out_dir
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

    if not args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    mfl = multiclass_focal_loss()

    custom_objects = {'mb':masked_binary_xent, 'multiclass_acc':multiclass_acc,
            'binary_acc':binary_acc, 'masked_categorical_xent':masked_categorical_xent,
            'multiclass_FL':mfl}
    model_paths = args.model
    if args.evaluate_all_mt:
        for path, row in irrigated_path_rows_mt():
            image_directory = args.image_dir + "_".join([str(path), str(row), str(args.year)])
            outfile = args.outfile
            if args.include_path_row:
                outfile = os.path.splitext(outfile)[0]
                outfile = os.path.basename(os.path.normpath(image_directory)) + outfile + ".tif"
            outfile = os.path.join(out_dir, outfile)
            evaluate_image_many_shot(image_directory, 
                     model_paths=model_paths, 
                     n_classes=args.n_classes,
                     n_overlaps=1,
                     outfile=outfile,
                     custom_objects=custom_objects)
            image_directory = args.image_dir
    else:
        outfile = args.outfile
        if args.include_path_row:
            outfile = os.path.splitext(outfile)[0]
            outfile = os.path.basename(os.path.normpath(args.image_dir)) + outfile + ".tif"
        outfile = os.path.join(out_dir, outfile)

        evaluate_image_many_shot(args.image_dir, 
                 model_paths=model_paths, 
                 n_classes=args.n_classes,
                 n_overlaps=1,
                 outfile=outfile,
                 custom_objects=custom_objects,
                 preprocessing_func=args.preprocessing_func)
