import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import keras.backend as K
#tf.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import geopandas as gpd
import sys
from glob import glob
from skimage import transform, util
from tensorflow.keras.layers import (Conv2D, Input, MaxPooling2D, Conv2DTranspose, 
Concatenate, Dropout, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from multiprocessing import Pool
from rasterio import open as rasopen
from rasterio.mask import mask
from shapely.geometry import shape
from fiona import open as fopen
from data_generators import generate_training_data, load_raster, preprocess_data

NO_DATA = -1
MAX_POOLS = 3
CHUNK_SIZE = 1248 # some value that is evenly divisible by 2^3.
NUM_CLASSES = 4
WRS2 = '../spatial_data/wrs2_descending_usa.shp'

def custom_objective(y_true, y_pred):
    '''I want to mask all values that 
       are not data, given a y_true 
       that has NODATA values. The boolean mask 
       operation is failing. It should output
       a Tensor of shape (M, N_CLASSES), but instead outputs a (M, )
       tensor.'''
    # fig, ax = plt.subplots(ncols=2)
    # boo = y_true.numpy()
    # boo2 = y_pred.numpy()
    # ax[0].imshow(boo[0, :, :, 0])
    # ax[1].imshow(boo2[0, :, :, 0])
    # plt.show()
    # Let's see the argmax of y_true, y_pred.
    y_true = tf.reshape(y_true, (K.shape(y_true)[1]*K.shape(y_true)[2], NUM_CLASSES))
    y_pred = tf.reshape(y_pred, (K.shape(y_pred)[1]*K.shape(y_pred)[2], NUM_CLASSES))
    masked = tf.not_equal(y_true, NO_DATA)
    indices = tf.where(masked)
    indices = tf.to_int32(indices)
    indices = tf.slice(indices, [0, 0], [K.shape(indices)[0], 1])
    y_true_mask = tf.gather_nd(params=y_true, indices=indices)
    y_pred_mask = tf.gather_nd(params=y_pred, indices=indices)
    return tf.keras.losses.categorical_crossentropy(y_true_mask, y_pred_mask)

def fcnn_functional(n_classes):

    x = Input((None, None, 36))

    c1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
    c1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(c1)
    mp1 = MaxPooling2D(pool_size=2, strides=(2, 2))(c1)

    c2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(mp1)
    c2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(c2)
    mp2 = MaxPooling2D(pool_size=2, strides=(2, 2))(c2)
    mp2 = Dropout(0.5)(mp2)

    c3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(mp2)
    c3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(c3)
    mp3 = MaxPooling2D(pool_size=2, strides=(2, 2))(c3)
    
    last_conv = Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(mp3)

    u1 = UpSampling2D(size=(2, 2))(last_conv)
    u1 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(u1)
    u1 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(u1)

    u1_c3 = Concatenate()([c3, u1])

    u2 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(u1_c3)
    u2 = UpSampling2D(size=(2, 2))(u2)
    u2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(u2)
    u2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(u2)
    u2 = Dropout(0.5)(u2)

    u2_c2 = Concatenate()([u2, c2])
    u2_c2 = Dropout(0.5)(u2_c2)

    c4 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(u2_c2)
    u3 = UpSampling2D(size=(2, 2))(c4)
    u3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(u3)

    u3_c1 = Concatenate()([u3, c1])

    c5 = Conv2D(filters=n_classes, kernel_size=(3,3), activation='softmax', padding='same')(u3_c1)

    model = Model(inputs=x, outputs=c5) 
    #model.summary()
    return model

def fcnn_model(n_classes):
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=8, padding='same', activation='relu',
        input_shape=(None, None, 36)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=4, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(filters=n_classes, kernel_size=2, padding='same',
        activation='softmax')) # 1x1 convolutions for pixel-wise prediciton.
    # Take a look at the model summary
    #model.summary()
    return model


def evaluate_image(master_raster, model, outfile=None, ii=None):

    if not os.path.isfile(master_raster):
        print("Master raster not created for {}".format(suffix))
        # TODO: More extensive error handling.
    else:
        master, meta = load_raster(master_raster)
        
        class_mask = np.zeros((2, master.shape[1], master.shape[2]))
        out = np.zeros((NUM_CLASSES, master.shape[2], master.shape[1]))
        x_plot =[]
        y_plot =[]
        for i in range(0, master.shape[1], CHUNK_SIZE):
            for j in range(0, master.shape[2], CHUNK_SIZE):
                sub_master = master[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                sub_mask = class_mask[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                sub_master, sub_mask, cut_rows, cut_cols = preprocess_data(sub_master, sub_mask, return_cuts=True)
                preds = model.predict(sub_master)
                preds = preds[0, :, :, :]
                preds = np.swapaxes(preds, 1, 2)
                preds = np.swapaxes(preds, 0, 1)
                #preds = np.argmax(preds, axis=2)

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

            sys.stdout.write("ii: {} Percent done: {:.4f}\r".format(ii, i / master.shape[1]))

    out = np.swapaxes(out, 1, 2)
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
    model = create_model(NUM_CLASSES)
    tb = TensorBoard(log_dir='graphs/')
    train_generator = generate_training_data(training_directory, box_size)
    model.fit_generator(train_generator, 
            steps_per_epoch=steps_per_epoch, 
            epochs=epochs,
            verbose=1,
            callbacks=[tb],
            use_multiprocessing=True)
    return model

def create_model(n_classes):
    model = fcnn_functional(n_classes)
    model.compile(loss=custom_objective,
                 optimizer=tf.keras.optimizers.Adam())
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
    # Steps:
    # 1. split shapefiles
    # 2. download all images 
    # 3. Get all shapefiles corresponding to a given
    # p/r. If there aren't n_classes shapefiles
    # for a given path/row, randomly sample from
    # places outside the annotated data that is present.
    # else: generate a class mask with class balance, 
    # where all of the pixels are drawn from irrigated
    # areas and only some from other areas.
    # 4. Input this into the model.
    # Here assume steps 3 and 4 are done and then synthesize
    # the steps into one coherent file.
    # need an easier way to specify year.
    # Let's do a binary classification model.
    shapefile_directory = 'shapefile_data/backup'
    sample_dir = os.path.join(shapefile_directory, 'sample_points')
    image_directory = 'master_rasters/'
    target = 'irrigated'
    fallow = 'Fallow'
    forest = 'Forrest'
    other = 'other'
    target_dict = {target:0, fallow:1, forest:2, other:3}
    year = 2013
    done = set()
    train_raster = 'master_raster_'
    mask_raster = 'class_mask_'
    n_classes = 2
    out_directory = 'evaluated_images_fully_conv/'
    training_directory = 'training_data'

    m_dir = 'data_from_disk/balance/all_classes' 
    pth = os.path.join(m_dir, "model_adam30epochs.h5")
    if not os.path.isfile(pth):
        model = train_model(training_directory, 87, epochs=30)
        model.save(pth)
    else:
        model = tf.keras.models.load_model(pth,
                custom_objects={'custom_objective':custom_objective})
    ii = 0
    for f in glob(os.path.join(image_directory, "*.tif")):
        if "class" not  in f:
            out = os.path.basename(f)
            os.path.split(out)[1]
            out = out[out.find("_")+1:]
            out = out[out.find("_"):]
            out = os.path.splitext(out)[0]
            out = '30epochstestadamevaluated' + out + ".tif"
            #out = 'testing' + out + ".tif"
            out = os.path.join(m_dir, out)
            ii += 1
            evaluate_image(f, model, outfile=out, ii=ii)
    clip_rasters(m_dir, "test")

