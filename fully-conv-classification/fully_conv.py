import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras.backend as K
import tensorflow as tf
#tf.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import json
import geopandas as gpd
import sys
from glob import glob
from skimage import transform, util
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from rasterio import open as rasopen
from rasterio.mask import mask
from shapely.geometry import shape
from fiona import open as fopen
from data_generators import generate_training_data, load_raster, preprocess_data
from data_utils import generate_class_mask
from models import fcnn_functional, fcnn_model, fcnn_functional_small

NO_DATA = -1
CHUNK_SIZE = 608 # some value that is divisible by 2^MAX_POOLS.
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

def custom_objective_binary(y_true, y_pred):
    '''I want to mask all values that 
       are not data, given a y_true 
       that has NODATA values. The boolean mask 
       operation is failing. It should output
       a Tensor of shape (M, N_CLASSES), but instead outputs a (M, )
       tensor.'''
    y_true = tf.reshape(y_true, (K.shape(y_true)[1]*K.shape(y_true)[2], NUM_CLASSES))
    y_pred = tf.reshape(y_pred, (K.shape(y_pred)[1]*K.shape(y_pred)[2], NUM_CLASSES))
    masked = tf.not_equal(y_true, NO_DATA)
    indices = tf.where(masked)
    indices = tf.to_int32(indices)
    indices = tf.slice(indices, [0, 0], [K.shape(indices)[0], 1])
    y_true_masked = tf.gather_nd(params=y_true, indices=indices)
    y_pred_masked = tf.gather_nd(params=y_pred, indices=indices)
    return tf.keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)

def custom_objective(y_true, y_pred):
    '''I want to mask all values that 
       are not data, given a y_true 
       that has NODATA values. The boolean mask 
       operation is failing. It should output
       a Tensor of shape (M, N_CLASSES), but instead outputs a (M, )
       tensor.'''
    y_true = tf.reshape(y_true, (K.shape(y_true)[1]*K.shape(y_true)[2], NUM_CLASSES))
    y_pred = tf.reshape(y_pred, (K.shape(y_pred)[1]*K.shape(y_pred)[2], NUM_CLASSES))
    masked = tf.not_equal(y_true, NO_DATA)
    indices = tf.where(masked)
    indices = tf.to_int32(indices)
    indices = tf.slice(indices, [0, 0], [K.shape(indices)[0], 1])
    y_true_masked = tf.gather_nd(params=y_true, indices=indices)
    y_pred_masked = tf.gather_nd(params=y_pred, indices=indices)
    return tf.keras.losses.categorical_crossentropy(y_true_masked, y_pred_masked)


def evaluate_image(master_raster, model, max_pools, outfile=None, ii=None):

    if not os.path.isfile(master_raster):
        print("Master raster not created for {}".format(suffix))
        # TODO: More extensive handling of this case.
    else:
        master, meta = load_raster(master_raster)
        class_mask = np.zeros((2, master.shape[1], master.shape[2])) # Just a placeholder
        out = np.zeros((master.shape[2], master.shape[1], NUM_CLASSES))

        for i in range(0, master.shape[1], CHUNK_SIZE):
            for j in range(0, master.shape[2], CHUNK_SIZE):
                sub_master = master[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                sub_mask = class_mask[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                sub_master, sub_mask, cut_rows, cut_cols = preprocess_data(sub_master, sub_mask,
                        max_pools, return_cuts=True)
                preds = model.predict(sub_master)
                preds = preds[0, :, :, :]

                if cut_cols == 0 and cut_rows == 0:
                    out[j:j+CHUNK_SIZE, i:i+CHUNK_SIZE, :] = preds
                elif cut_cols == 0 and cut_rows != 0:
                    ofs = master.shape[1]-cut_rows
                    out[j:j+CHUNK_SIZE, i:ofs, :] = preds
                elif cut_cols != 0 and cut_rows == 0:
                    ofs = master.shape[2]-cut_cols
                    out[j:ofs, i:i+CHUNK_SIZE, :] = preds
                elif cut_cols != 0 and cut_rows != 0:
                    ofs_col = master.shape[2]-cut_cols
                    ofs_row = master.shape[1]-cut_rows
                    out[j:ofs_col, i:ofs_row, :] = preds
                else:
                    print("whatcha got goin on here?")

            sys.stdout.write("N eval: {}. Percent done: {:.4f}\r".format(ii, i / master.shape[1]))

    out = np.swapaxes(out, 0, 2)
    out = out.astype(np.float32)
    if outfile:
        save_raster(out, outfile, meta)
    return out

def save_raster(arr, outfile, meta, count=NUM_CLASSES):
    meta.update(count=count)
    with rasopen(outfile, 'w', **meta) as dst:
        dst.write(arr)


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

    if outfile:
        save_raster(out_image, outfile, meta)

def clip_rasters(evaluated_tif_dir, include_string):
    for f in glob(os.path.join(evaluated_tif_dir, "*.tif")):
        if include_string in f:
            out = os.path.basename(f)
            out = out[out.find("_")+1:]
            out = out[out.find("_")+1:]
            out = out[out.find("_")+1:]
            path = out[:2]
            row = out[3:5]
            clip_raster(f, int(path), int(row), outfile=f)

def evaluate_images(image_directory, include_string, max_pools, exclude_string, prefix, save_dir):
    ii = 0
    for f in glob(os.path.join(image_directory, "*.tif")):
        if exclude_string not in f and include_string in f:
            out = os.path.basename(f)
            os.path.split(out)[1]
            out = out[out.find("_"):]
            out = os.path.splitext(out)[0]
            out = prefix + out + ".tif"
            out = os.path.join(save_dir, out)
            ii += 1
            evaluate_image(f, model, max_pools=max_pools, outfile=out, ii=ii)

def compute_iou(y_pred, y_true):
     ''' This is slow. '''
     y_pred = y_pred.flatten()
     y_true = y_true.flatten()
     current = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
     print(current)
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return np.mean(IoU)

def get_iou():
    shpfiles = [
    'shapefile_data/test/MT_Huntley_Main_2013_372837_28.shp',
    'shapefile_data/test/MT_FLU_2017_Fallow_372837_28.shp',
    'shapefile_data/test/MT_FLU_2017_Forrest_372837_28.shp',
    'shapefile_data/test/MT_other_372837_28.shp']

    m_dir = 'eval_test/all_ims/'
    ls = []
    mask = image_directory + 'class_mask_37_28_2013.tif'
    for f in shpfiles:
        msk = generate_class_mask(f, mask)
        msk[msk != NO_DATA] = 1 
        ls.append(msk)
    y_true = np.vstack(ls)
    indices = np.where(y_true != NO_DATA)
    y_true = y_true[:, indices[1], indices[2]]
    y_true = np.argmax(y_true, axis=0)
    for f in glob(m_dir + "*.tif"):
        y_pred, meta = load_raster(f)
        y_pred = y_pred[:, indices[1], indices[2]]
        y_pred = np.round(y_pred)
        y_pred.astype(np.int32)
        y_pred = np.argmax(y_pred, axis=0)
        print(f, compute_iou(y_pred, y_true))

def train_model(training_directory, model, steps_per_epoch, valid_steps, max_pools, box_size=0, epochs=3):
    ''' This function assumes that train/test data are
    subdirectories of training_directory, with
    the names train/test.'''
    model = model(NUM_CLASSES)
    if NUM_CLASSES <= 2:
        model.compile(loss=custom_objective_binary,
                     metrics=[m_acc],
                     optimizer='adam')
    else:
        model.compile(loss=custom_objective,
                 metrics=['accuracy', m_acc],
                 optimizer=tf.keras.optimizers.Adam(lr=0.0001))

    tb = TensorBoard(log_dir='graphs/30epochssimple/')
    train = os.path.join(training_directory, 'train')
    test = os.path.join(training_directory, 'test')
    train_generator = generate_training_data(train, max_pools, random_sample=False,
            train=True, box_size=box_size)
    test_generator = generate_training_data(test, max_pools, random_sample=False,
            train=False, box_size=box_size)
    model.fit_generator(train_generator, 
            steps_per_epoch=steps_per_epoch, 
            epochs=epochs,
            verbose=1,
            callbacks=[tb],
            class_weight=[31.0, 1, 2.16, 67.76],
            use_multiprocessing=True)
            #validation_data=test_generator,
           #validation_steps=valid_steps,

    return model

if __name__ == '__main__':

    image_directory = 'master_rasters/'
    training_directory = 'training_data/multiclass/'
    m_dir = 'eval_test/multiclass/' 

    #get_iou()
    # models = [fcnn_functional, fcnn_functional_small, fcnn_model]
    # save_dirs = [os.path.join(m_dir, "complex_fcnn"), os.path.join(m_dir, "simple_fcnn"), 
    #         os.path.join(m_dir, 'no_pools')]
    # model_names = ["multiclass_complex_fcnn.h5", 'multiclass_simple_fcnn.h5',
    #         'multiclass_no_pools_fcnn.h5']
    # raster_names = ["complex_fcnnmulticlass", "simple_fcnnmulticlass", "no_poolsmulticlass"]

    models = [fcnn_functional]
    save_dirs = [os.path.join(m_dir, "augmented/")]
    model_names = ["complex_fcnn_augmented.h5"]
    raster_names = ["class_weightscomplexaugmented"]
    i = 1
    max_pools = 5
    for model_func, save_dir, model_name, raster_name in zip(models, save_dirs, model_names, raster_names):
        pth = os.path.join(save_dir, model_name)
        if not os.path.isfile(pth):
            model = train_model(training_directory, model_func, steps_per_epoch=764,
                    valid_steps=246, max_pools=max_pools, epochs=5)
            model.save(pth)
        else:
            model = tf.keras.models.load_model(pth,
                    custom_objects={'m_acc':m_acc, 'custom_objective':custom_objective})

        evaluate_images(image_directory, include_string="37_28", 
                exclude_string="class", max_pools=max_pools, prefix=raster_name, save_dir=save_dir) 
        clip_rasters(save_dir, "37_28")
        if i == 2:
            max_pools = 3
        if i == 3:
            max_pools = 0
        i += 1
