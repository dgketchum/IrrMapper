import os
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import pdb
from sys import stdout
from tensorflow.keras.models import load_model
from glob import glob
from rasterio.errors import RasterioIOError

from data_utils import save_raster, stack_rasters, paths_map, load_raster, clip_raster
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
        image[i, :, :][masked_image[0]==1] = np.nan
    meta.update(count=image.shape[0])
    meta.update(nodata=np.nan)
    return image, meta
    

def evaluate_image_many_shot(path, row, year, image_directory, model, num_classes=4, n_overlaps=4, outfile=None, ii=None):
    ''' To recover from same padding, slide many different patches over the image. '''
    suffix = '{}_{}_{}'.format(path, row, year) 
    image_path = os.path.join(image_directory, suffix)
    if not os.path.isdir(image_path):
        print('Images not downloaded for {}'.format(image_path))
        return
    paths_mapping = paths_map(image_path)
    try:
        _, meta = load_raster(paths_mapping['B1.TIF'][0])
        master = stack_rasters(paths_mapping, path, row, year)
    except Exception as e:
        print(e)
        return
    class_mask = np.ones((master.shape[1], master.shape[2], num_classes)) # Just a placeholder
    out = np.zeros((master.shape[2], master.shape[1], num_classes))
    chunk_size = 608
    diff = 608
    stride = 608
    overlap_step = 10
    for k in range(0, n_overlaps*overlap_step, overlap_step):
        for i in range(k, master.shape[1]-diff, stride):
            for j in range(k, master.shape[2]-diff, stride):
                sub_master = master[:, i:i+chunk_size, j:j+chunk_size]
                sub_mask = class_mask[i:i+chunk_size, j:j+chunk_size, :]
                sub_master = np.swapaxes(sub_master, 0, 2)
                sub_master = np.swapaxes(sub_master, 0, 1)
                sub_master = np.expand_dims(sub_master, 0)
                sub_mask = np.expand_dims(sub_mask, 0)
                preds = model.predict([sub_master, sub_mask]) 
                preds = np.exp(preds)
                soft = preds / np.sum(preds, axis=-1, keepdims=True)
                soft = np.swapaxes(soft, 1, 2)
                out[j:j+chunk_size, i:i+chunk_size, :] += soft[0]
            stdout.write("K: {} of {}. Percent done: {:.2f}\r".format(k // overlap_step + 1, n_overlaps, i / master.shape[1]))
    out = np.swapaxes(out, 0, 2)
    out = out.astype(np.float32)
    meta.update(dtype=np.float32)
    out /= n_overlaps
    if outfile:
        save_raster(out, outfile, meta, count=num_classes)
    return out

if __name__ == '__main__':
    mt_path = [42, 41, 40, 39, 38, 37, 36, 35, 42, 41, 40, 39, 38, 37, 36, 35, 41, 40, 39, 38, 37,
            36, 35, 34, 40, 39, 38, 37, 36, 35, 34]
    mt_row = [26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28,
            28, 28, 28, 29, 29, 29, 29, 29, 29, 29]
    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]

    landsat_directory = '/home/thomas/share/image_data/train/'
    save_directory = '/home/thomas/share/fmask_evaluated_mt/'
    for f in glob("/home/thomas/share/evaluated_mt/" + "*.tif"):
        _, path, row, year = os.path.basename(f).split('_')
        year = year[:-4]
        outfile = os.path.join(save_directory, os.path.basename(f))
        if not os.path.isfile(outfile):
            print(path, row, year)
            try:
                out, meta = fmask_evaluated_image(f, path, row, year, landsat_directory)
                save_raster(out, outfile, meta)
                clip_raster(outfile, int(path), int(row))
            except RasterioIOError as e:
                print(e)

    # n_classes = 5
    # model_name = 'augmentation_irr_and_wetlands_no_class_weights.h5'
    # image_directory = '/home/thomas/share/image_data/train/'
    # save_directory = '/home/thomas/share/evaluated_mt/'
    # model = load_model("models/" + model_name, custom_objects={'tf':tf, '_epsilon':_epsilon, 
    #     'weighted_loss':weighted_loss})
    # for year in years:
    #     for path, row in zip(mt_path, mt_row):
    #         print("Evaluating", path, row, year)
    #         suffix = 'irr_{}_{}_{}.tif'.format(path, row, year) 
    #         outfile = os.path.join(save_directory, suffix)
    #         if not os.path.isfile(outfile):
    #             try:
    #                 evaluate_image_many_shot(path, row, year, image_directory,
    #                     model, outfile=outfile, num_classes=n_classes)
    #             except Exception as e:
    #                 print(e)
    #                 continue
    #         else:
    #             print("Image {} already exists.".format(outfile))
