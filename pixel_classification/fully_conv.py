import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import keras.backend as K
#tf.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import json
import geopandas as gpd
import sys
from glob import glob
from skimage import transform, util
from tensorflow.keras.callbacks import TensorBoard
from rasterio import open as rasopen
from rasterio.mask import mask
from shapely.geometry import shape
from fiona import open as fopen
from data_generators import generate_training_data, load_raster, preprocess_data
from models import fcnn_functional

NO_DATA = -1
MAX_POOLS = 5
CHUNK_SIZE = 1248 # some value that is divisible by 2^3.
NUM_CLASSES = 4
WRS2 = '../spatial_data/wrs2_descending_usa.shp'

def m_acc(y_true, y_pred):
    ''' Calculate accuracy from masked data.
    The built-in accuracy metric uses all data (masked & unmasked).'''
    y_true = tf.reshape(y_true, (K.shape(y_true)[1]*K.shape(y_true)[2], NUM_CLASSES))
    y_pred = tf.reshape(y_pred, (K.shape(y_pred)[1]*K.shape(y_pred)[2], NUM_CLASSES))
    masked = tf.not_equal(y_true, NO_DATA)
    indices = tf.where(masked)
    indices = tf.to_int32(indices)
    indices = tf.slice(indices, [0, 0], [K.shape(indices)[0], 1])
    y_true_masked = tf.gather_nd(params=y_true, indices=indices)
    y_pred_masked = tf.gather_nd(params=y_pred, indices=indices)
    return K.cast(K.equal(K.argmax(y_true_masked, axis=-1), K.argmax(y_pred_masked, axis=-1)), K.floatx())

def custom_objective(y_true, y_pred):
    '''I want to mask all values that 
       are not data, given a y_true 
       that has NODATA values. The boolean mask 
       operation is failing. It should output
       a Tensor of shape (M, N_CLASSES), but instead outputs a (M, )
       tensor.'''
    # Dice coefficient?
    y_true = tf.reshape(y_true, (K.shape(y_true)[1]*K.shape(y_true)[2], NUM_CLASSES))
    y_pred = tf.reshape(y_pred, (K.shape(y_pred)[1]*K.shape(y_pred)[2], NUM_CLASSES))
    masked = tf.not_equal(y_true, NO_DATA)
    indices = tf.where(masked)
    indices = tf.to_int32(indices)
    indices = tf.slice(indices, [0, 0], [K.shape(indices)[0], 1])
    y_true_masked = tf.gather_nd(params=y_true, indices=indices)
    y_pred_masked = tf.gather_nd(params=y_pred, indices=indices)
    return tf.keras.losses.categorical_crossentropy(y_true_masked, y_pred_masked)

def evaluate_image(master_raster, model, outfile=None, ii=None):

    if not os.path.isfile(master_raster):
        print("Master raster not created for {}".format(suffix))
        # TODO: More extensive handling of this case.
    else:
        master, meta = load_raster(master_raster)
        class_mask = np.zeros((2, master.shape[1], master.shape[2])) # Just a placeholder
        out = np.zeros((master.shape[1], master.shape[2], NUM_CLASSES))

        for i in range(0, master.shape[1], CHUNK_SIZE):
            for j in range(0, master.shape[2], CHUNK_SIZE):
                sub_master = master[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                sub_mask = class_mask[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                sub_master, sub_mask, cut_rows, cut_cols = preprocess_data(sub_master, sub_mask, return_cuts=True)
                preds = model.predict(sub_master)
                preds = preds[0, :, :, :]

                if cut_cols == 0 and cut_rows == 0:
                    out[:,j:j+CHUNK_SIZE, i:i+CHUNK_SIZE] = preds
                elif cut_cols == 0 and cut_rows != 0:
                    ofs = master.shape[1]-cut_rows
                    out[:, j:j+CHUNK_SIZE, i:ofs] = preds
                elif cut_cols != 0 and cut_rows == 0:
                    ofs = master.shape[2]-cut_cols
                    out[:, j:ofs, i:i+CHUNK_SIZE] = preds
                elif cut_cols != 0 and cut_rows != 0:
                    ofs_col = master.shape[2]-cut_cols
                    ofs_row = master.shape[1]-cut_rows
                    out[:, j:ofs_col, i:ofs_row] = preds
                else:
                    print("whatcha got goin on here?")

            sys.stdout.write("N eval: {}. Percent done: {:.4f}\r".format(ii, i / master.shape[1]))

    out = np.swapaxes(out, 0, 2)
    out = out.astype(np.float32)
    if outfile:
        save_raster(out, outfile, meta)
    return out

def save_raster(arr, outfile, meta):
    meta.update(count=NUM_CLASSES)
    with rasopen(outfile, 'w', **meta) as dst:
        dst.write(arr)

def train_model(training_directory, steps_per_epoch, box_size=0, epochs=3):
    # image shape will change here, so it must be
    # inferred at runtime.
    ''' This function assumes that train/test data are
    subdirectories of training_directory, with
    the names train/test.'''
    model = create_model(NUM_CLASSES)
    tb = TensorBoard(log_dir='graphs/')
    train = os.path.join(training_directory, 'train')
    test = os.path.join(training_directory, 'test')
    train_generator = generate_training_data(train, box_size)
    test_generator = generate_training_data(test, box_size)
    model.fit_generator(train_generator, 
            validation_data=test_generator,
            validation_steps=31,
            steps_per_epoch=steps_per_epoch, 
            epochs=epochs,
            verbose=1,
            callbacks=[tb],
            use_multiprocessing=True)
    return model


def create_model(n_classes):
    model = fcnn_functional(n_classes)
    model.compile(loss=custom_objective,
                 metrics=[m_acc],
                 optimizer='adam')
    return model


def get_features(gdf, path, row):
    tmp = json.loads(gdf.to_json())
    features = []
    for feature in tmp['features']:
        if feature['properties']['PATH'] == path and feature['properties']['ROW'] == row: 
            features.append(feature['geometry'])
    return features

def clip_raster(evaluated, path, row, outfile=None):

    shp = gpd.read_file(WRS2)

    with rasopen(evaluated, 'r') as src:
        shp = shp.to_crs(src.crs)
        meta = src.meta.copy()
        features = get_features(shp, path, row)
        out_image, out_transform = mask(src, shapes=features, nodata=np.nan)

    #out_image[out_image != 0] = np.nan
    if outfile:
        save_raster(out_image, outfile, meta)

def clip_rasters(evaluated_tif_dir, include_string):
    for f in glob(os.path.join(evaluated_tif_dir, "*.tif")):
        if include_string in f:
            out = os.path.basename(f)
            os.path.split(out)[1]
            out = out[out.find("_")+1:]
            #out = out[out.find("_")+1:]
            path = out[:2]
            row = out[3:5]
            clip_raster(f, int(path), int(row), outfile=f)

# TODO: Implement IoU so I can actually see how my model is doing.

if __name__ == '__main__':

    shapefile_directory = 'shapefile_data/backup'
    image_directory = 'master_rasters/'
    training_directory = 'training_data'

    m_dir = 'eval_test' 
    pth = os.path.join(m_dir, "model_acctst.h5")
    if not os.path.isfile(pth):
        model = train_model(training_directory, 109, epochs=2)
        model.save(pth)
    else:
        model = tf.keras.models.load_model(pth,
                custom_objects={'custom_objective':custom_objective})
    ii = 0
    for f in glob(os.path.join(image_directory, "*.tif")):
        if "class" not  in f and "37_28" in f:
            out = os.path.basename(f)
            os.path.split(out)[1]
            out = out[out.find("_")+1:]
            out = out[out.find("_"):]
            out = os.path.splitext(out)[0]
            out = 'complexfcnn_multiclass' + out + ".tif"
            #out = 'testing' + out + ".tif"
            out = os.path.join(m_dir, out)
            ii += 1
            evaluate_image(f, model, outfile=out, ii=ii)
    clip_rasters(m_dir, "37_28")

