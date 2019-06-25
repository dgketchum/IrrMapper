import os
import numpy as np
import tensorflow as tf
from data_utils import clip_rasters, save_raster, stack_rasters, paths_map
from sys import stdout
from tensorflow.keras.models import load_model
from data_generators import load_raster
import matplotlib.pyplot
import keras.backend as K
from fully_conv import weighted_loss, weighted_focal_loss

_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)

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
    n_classes = 5
    model_name = 'augmentation_irr_and_wetlands_no_class_weights.h5'
    image_directory = '/home/thomas/share/image_data/train/'
    save_directory = '/home/thomas/share/evaluated_mt/'
    model = load_model("models/" + model_name, custom_objects={'tf':tf, '_epsilon':_epsilon, 
        'weighted_loss':weighted_loss})
    for year in years:
        for path, row in zip(mt_path, mt_row):
            print("Evaluating", path, row, year)
            suffix = 'irr_{}_{}_{}.tif'.format(path, row, year) 
            outfile = os.path.join(save_directory, suffix)
            if not os.path.isfile(outfile):
                evaluate_image_many_shot(path, row, year, image_directory,
                    model, outfile=outfile, num_classes=n_classes)
            else:
                print("Image {} already exists.".format(outfile))
