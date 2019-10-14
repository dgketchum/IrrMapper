import os
#E os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import numpy as np
import keras.backend as K
import tensorflow as tf
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

from data_utils import (save_raster, stack_rasters, stack_rasters_multiprocess, paths_map_multiple_scenes, load_raster, clip_raster, paths_mapping_single_scene)
from losses import multiclass_acc, masked_binary_xent, dice_loss, binary_acc, binary_focal_loss
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
                preds = expit(preds[0])
                out[i:i+chunk_size, j:j+chunk_size, :] += preds
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
        n_overlaps=4, outfile=None, custom_objects=None):
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
    out_arr = np.zeros((1, image_stack.shape[1], image_stack.shape[2]))
    for i, model_path in enumerate(model_paths):
        print('loading {}'.format(model_path))
        model = load_model(model_path, custom_objects=custom_objects)
        out_arr += _evaluate_image_return_logits(model, image_stack, n_classes=n_classes,
            n_overlaps=n_overlaps)
        del model

    print(out_arr.shape)
    temp_mask = np.zeros((1, out_arr.shape[1], out_arr.shape[2]))
    fmasked_image = concatenate_fmasks(image_directory, temp_mask, meta, nodata=1)
    # for i in range(out_arr.shape[0]):
    #     out_arr[i, :, :][fmasked_image.mask[0]] = np.nan
    meta.update(dtype=np.float64)
    out_arr /= n_overlaps
    if outfile:
        save_raster(out_arr, outfile, meta, count=n_classes)
    return out_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-i', '--image-dir', type=str, required=True)
    parser.add_argument('-o', '--out-dir', type=str)
    parser.add_argument('-n', '--n-classes', type=int, default=5)
    parser.add_argument('-b', '--binary', action='store_true')
    args = parser.parse_args()
    if args.out_dir is None:
        out_dir = os.path.dirname(os.path.splitext(args.model)[0])
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
    else: 
        out_dir = args.out_dir

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    custom_objects = {'mb':masked_binary_xent, 'binary_acc':binary_acc}

    # model_paths = glob('/home/thomas/IrrMapper/fully-conv-classification/ensemble_models/test3/*.h5')
    # model_paths = sorted(model_paths)
    # model_paths = model_paths[len(model_paths)-1]
    model_paths = args.model
    image_directory = args.image_dir
    outfile = os.path.join(os.path.basename(os.path.normpath(image_directory)) +
            '_random_majority_sample.tif')
    outfile = os.path.join(out_dir, outfile)
    evaluate_image_many_shot(image_directory, 
             model_paths=model_paths, 
             n_classes=args.n_classes,
             n_overlaps=1,
             outfile=outfile,
             custom_objects=custom_objects)
