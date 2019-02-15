import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from glob import glob
from skimage import transform, util
from tensorflow.keras.layers import (Conv2D, Input, MaxPooling2D, Conv2DTranspose, 
Concatenate, Dropout, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from data_utils import generate_class_mask, get_shapefile_path_row
from multiprocessing import Pool
from rasterio import open as rasopen

NO_DATA = -1

def custom_objective(y_true, y_pred):
    '''I want to mask all values that 
       are not data, given a y_true 
       that has NODATA values. '''
    y_true = tf.reshape(y_true, (1080*1920, 2))
    y_pred = tf.reshape(y_pred, (1080*1920, 2))
    masked = tf.not_equal(y_true, NO_DATA)
    y_true_mask = tf.boolean_mask(y_true, masked)
    y_pred_mask = tf.boolean_mask(y_pred, masked)
    return tf.keras.losses.binary_crossentropy(y_true_mask, y_pred_mask)

def fcnn_functional(image_shape, n_classes):

    x = Input(image_shape)

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
    model.summary()
    return model

def one_hot_encoding(class_mask, n_classes):
    '''Assumes classes range from 0 -> (n-1)'''
    shp = class_mask.shape
    out = np.ones((shp[0], shp[1], n_classes))*NO_DATA
    for i in range(n_classes):
        out[:, :, i][class_mask == i] = 1
    return out

def rotation(image, angle):
    return transform.rotate(image, angle, mode='constant', cval=NO_DATA)

def random_noise(image):
    return util.random_noise(image)

def h_flip(image):
    return image[:, ::-1]

def augment_data(image, class_mask):
    '''Randomly augments an image.'''
    if np.random.randint(2):
        deg = np.random.uniform(-25, 25)
        image = rotation(image, deg)
        class_mask = rotation(class_mask, deg)
    if np.random.randint(2):
        image = random_noise(image)
    if np.random.randint(2):
        image = h_flip(image)
        class_mask = h_flip(class_mask)
    if np.random.randint(2):
        image = np.flipud(image)
        class_mask = np.flipud(class_mask)
    return image, class_mask

def generate(image_directory, box_size):
    while True:
        for f in glob(image_directory + "*.json"):
            jpg = f[:-13] + ".jpg"
            class_mask, input_image = generate_class_mask(f, jpg, box_size=box_size)
            if class_mask is None:
                continue
            if np.random.randint(2): 
                input_image, class_mask = augment_data(input_image, class_mask)

            X, y = preprocess_training_data(input_image, class_mask)

            yield X, y

def create_model(image_shape, n_classes):
    model = fcnn_functional(image_shape, n_classes)
    model.compile(loss=custom_objective,
                 optimizer='adam', 
                 metrics=['accuracy'])
    return model

def train_model(train_directory, test_directory, image_shape, box_size=6, epochs=15):
    n_classes = 2
    model = create_model(image_shape, n_classes)
    tb = TensorBoard(log_dir='graphs/')
    n_augmented = 0
    train_generator = generate(train_directory, box_size)
    test_generator = generate(test_directory, box_size)
    model.fit_generator(train_generator, 
            steps_per_epoch=50, 
            epochs=epochs,
            verbose=1,
            callbacks=[tb],
            validation_data=test_generator,
            validation_steps=4,
            use_multiprocessing=True)
    return model

def all_matching_shapefiles(to_match, shapefile_directory):
    out = []
    pr = get_shapefile_path_row(to_match)
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if get_shapefile_path_row(f) == pr and to_match not in f:
            out.append(f)
    return out

def generate_binary_train(shapefile_directory, image_directory, box_size):

    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if target in f:
            all_matches = all_matching_shapefiles(f, shapefile_directory)
            p, r = get_shapefile_path_row(f)
            suffix = '{}_{}_{}.tif'.format(p, r, year)
            master_raster = os.path.join(image_directory, train_raster + suffix)
            mask_file = os.path.join(image_directory, mask_raster + suffix)
            if not os.path.isfile(master_raster):
                print("Master raster not created for {}".format(suffix))
                # TODO: More extensive error handling.
            else:
                target_mask = generate_class_mask(f, mask_file)
                class_mask = np.ones((n_classes, target_mask.shape[1], target_mask.shape[2]))*NO_DATA
                class_mask[1, :, :] = target_mask

                required_instances = len(np.where(target_mask != NO_DATA)[0]) // (box_size*len(all_matches))
                masks = []
                for match in all_matches:
                    msk = generate_class_mask(match, mask_file)
                    samp = random_sample(msk, required_instances, box_size)
                    masks.append(samp)

                for i, s in enumerate(masks):
                    class_mask[0, :, :][s[0, :, :] != NO_DATA] = 1
                # May need to do some preprocessing.    
                yield class_mask, load_raster(master_raster)

def random_sample(class_mask, n_instances, box_size, class_code=0):
    out = np.where(class_mask != NO_DATA)
    # returns (elements from class_mask, indices_x, indices_y)
    out_x = out[1]
    out_y = out[2] 
    indices = np.random.choice(len(out_x), size=n_instances, replace=False)
    out_x = out_x[indices]
    out_y = out_y[indices] 
    class_mask[:, :, :] = NO_DATA
    if box_size == 0:
        class_mask[:, out_x, out_y] = class_code
    else:
        ofs = box_size // 2
        for x, y in zip(out_x, out_y):
            class_mask[0, x-ofs:x+ofs+1, y-ofs:y+ofs+1] = class_code
    return class_mask

def load_raster(master_raster):
    with rasopen(master_raster, 'r') as src:
        arr = src.read()
    return arr

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
    image_directory = 'master_rasters'
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
    box_size = 6


    for i, j in generate_binary_train(shapefile_directory, image_directory, box_size):
                           
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(i[0, :, :])
        ax[1].imshow(i[1, :, :])
        plt.show()
        

