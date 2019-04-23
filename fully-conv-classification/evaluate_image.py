import os
import numpy as np
import tensorflow as tf
from data_utils import clip_rasters, save_raster
from sys import stdout
from tensorflow.keras.models import load_model
from data_generators import load_raster
import matplotlib.pyplot
import keras.backend as K
from fully_conv import weighted_loss, weighted_focal_loss

_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)


def evaluate_image(master_raster, model, num_classes=4, outfile=None, ii=None):

    if not os.path.isfile(master_raster):
        print("Master raster not created for {}".format(suffix))
        # TODO: More extensive handling of this case.
    else:
        master, meta = load_raster(master_raster)
        class_mask = np.ones((master.shape[1], master.shape[2], num_classes)) # Just a placeholder
        out = np.zeros((master.shape[2], master.shape[1], num_classes))
        chunk_size = 608
        diff = 608
        stride = 608
        for i in range(0, master.shape[1]-diff, stride):
            for j in range(0, master.shape[2]-diff, stride):
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
                out[j:j+chunk_size, i:i+chunk_size, :] = soft

            stdout.write("N eval: {}. Percent done: {:.2f}\r".format(ii, i / master.shape[1]))

    out = np.swapaxes(out, 0, 2)
    out = out.astype(np.float32)
    if outfile:
        save_raster(out, outfile, meta, count=num_classes)
    return out

if __name__ == '__main__':
    master_raster_t = '/home/thomas/share/master_rasters/test/master_raster_37_28_2013.tif'
    master_raster = '/home/thomas/share/master_rasters/train/master_raster_39_27_2013.tif'
    model_name = 'normal_loss_50_irr_weight.h5'
    model = load_model("models/" + model_name, custom_objects={'tf':tf, '_epsilon':_epsilon, 
        'weighted_loss':weighted_loss})
    outfile = 'compare_model_outputs/new-feed-method/{}_39_27.tif'.format(model_name[:-3])
    evaluate_image(master_raster, model, outfile=outfile, num_classes=4)
    outfile = 'compare_model_outputs/new-feed-method/{}_37_28.tif'.format(model_name[:-3])
    evaluate_image(master_raster_t, model, outfile=outfile, num_classes=4)
