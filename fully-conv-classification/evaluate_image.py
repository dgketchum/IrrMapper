import os
import numpy as np
import keras.backend as K
import tensorflow as tf
import pdb
from sys import stdout
from tensorflow.keras.models import load_model
from glob import glob
from rasterio.errors import RasterioIOError
from matplotlib.pyplot import imshow, show, subplots
from multiprocessing import Pool

from data_utils import save_raster, stack_rasters, paths_map_multiple_scenes, load_raster, clip_raster
from fully_conv import weighted_loss, weighted_focal_loss
from data_generators import concatenate_fmasks


_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)

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
    

def evaluate_image_many_shot(path, row, year, image_directory, model_path, num_classes=4, n_overlaps=4, outfile=None, ii=None):
    ''' To recover from same padding, slide many different patches over the image. '''
    suffix = '{}_{}_{}'.format(path, row, year)
    image_path = os.path.join(image_directory, suffix)
    model = load_model(model_path, custom_objects={'weighted_loss':weighted_loss, 'tf':tf,
        '_epsilon':_epsilon})
    if not os.path.isdir(image_path):
        print('Images not downloaded for {}'.format(image_path))
        return
    paths_mapping = paths_map_multiple_scenes(image_path)
    try:
        template, meta = load_raster(paths_mapping['B1.TIF'][0])
        image_stack = stack_rasters(paths_mapping, meta, template.shape)
    except Exception as e:
        print(e)
        return
    class_mask = np.ones((1, image_stack.shape[2], image_stack.shape[1], num_classes)) # Just a placeholder
    out = np.zeros((image_stack.shape[2], image_stack.shape[1], num_classes))
    image_stack = np.swapaxes(image_stack, 0, 2)
    image_stack = np.expand_dims(image_stack, 0)
    imshow(image_stack[0, :, :, 9])
    show()
    chunk_size = 608
    diff = 608
    stride = 608
    overlap_step = 10
    for k in range(0, n_overlaps*overlap_step, overlap_step):
        for i in range(k, image_stack.shape[1]-diff, stride):
            for j in range(k, image_stack.shape[2]-diff, stride):
                sub_image_stack = image_stack[:, i:i+chunk_size, j:j+chunk_size, :]
                sub_mask = class_mask[:, i:i+chunk_size, j:j+chunk_size, :]
                preds = model.predict([sub_image_stack, sub_mask]) 
                preds = np.exp(preds)
                soft = preds / np.sum(preds, axis=-1, keepdims=True)
                out[i:i+chunk_size, j:j+chunk_size, :] += soft[0]
            stdout.write("K: {} of {}. Percent done: {:.2f}\r".format(k // overlap_step + 1, n_overlaps, i / image_stack.shape[1]))
    out = np.swapaxes(out, 0, 2)
    out = out.astype(np.float32)
    temp_mask = np.zeros((1, out.shape[1], out.shape[2]))
    masked_image = concatenate_fmasks(image_path, temp_mask, meta, nodata=1)
    for i in range(out.shape[0]):
        out[i, :, :][masked_image.mask[0]] = np.nan
    meta.update(dtype=np.float32)
    out /= n_overlaps
    if outfile:
        save_raster(out, outfile, meta, count=num_classes)
    return out


if __name__ == '__main__':

    path = 37
    row = 28
    year = 2013
    image_directory = "/home/thomas/share/image_data/train/"
    model_path = '/home/thomas/IrrMapper/fully-conv-classification/models/2019-03-08_40pacc_all_unit_weights/model.h5'
    evaluate_image_many_shot(path, row, year, image_directory, model_path, num_classes=6,
            n_overlaps=1, outfile='bad_accuracy.tif')
