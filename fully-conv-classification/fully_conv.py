import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import keras.backend as K
import tensorflow as tf
#tf.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
from sys import stdout
from glob import glob
from skimage import transform, util
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from rasterio import open as rasopen
from rasterio.mask import mask
from shapely.geometry import shape
from pickle import load as pload
from fiona import open as fopen
from data_generators import generate_training_data, load_raster, preprocess_data
from data_utils import clip_rasters, save_raster
from shapefile_utils import get_features, generate_class_mask
from models import (fcnn_functional, fcnn_model, fcnn_functional_small, unet, unet_weighted,
        weighted_unet_no_transpose_conv)

NO_DATA = -1
CHUNK_SIZE = 572 # some value that is divisible by 2^MAX_POOLS.
WRS2 = '../spatial_data/wrs2_descending_usa.shp'

def custom_objective(y_true, y_pred):
    y_true_for_loss = y_true
    mask = tf.not_equal(y_true, NO_DATA)
    y_true_for_loss = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_true_for_loss = tf.cast(y_true_for_loss, tf.int32)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true_for_loss)
    # the above line works in eager mode, but not otherwise.
    # losses = tf.keras.losses.sparse_categorical_crossentropy(y_true_for_loss, y_pred)
    out = tf.boolean_mask(losses, mask)
    return out


def weighted_loss(target, output):
    # All I need to do is multiply the output loss
    # by the weights that I input. 
    # Loss is of shape n_classesxwidthxheight
    # Weight map:
    # Raster of shape widthxheightx1, with weights
    # of zero where there is no data and weights of whatever the
    # correct weights are for all the other classes.
    return -tf.reduce_sum(target*output, len(output.get_shape())-1)


def masked_acc(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.argmax(y_pred, axis=3)
    mask = tf.not_equal(y_true, NO_DATA)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.cast(y_pred, tf.int64)
    return K.mean(tf.math.equal(y_true, y_pred))


def m_acc(y_true, y_pred):
    ''' Calculate accuracy from masked data.
    The built-in accuracy metric uses all data (masked & unmasked).'''
    mask = tf.not_equal(K.mean(y_pred, axis=len(y_pred.get_shape())-1), 0) 
    y_true = tf.math.argmax(y_true, axis=len(y_true.get_shape())-1)
    y_true_msk = tf.boolean_mask(y_true, mask)
    y_pred_exp = tf.math.exp(y_pred)
    y_pred_arg = tf.math.argmax(y_pred_exp, axis=len(y_pred_exp.get_shape())-1) 
    # so undo that operation with softmax (could also use tf.exp).
    y_pred_msk = tf.boolean_mask(y_pred_arg, mask) # mask the 0 values
    return K.mean(tf.math.equal(y_true_msk, y_pred_msk)) # get the accuracy


def evaluate_image_unet(master_raster, model, max_pools, channels='all', num_classes=4,
        outfile=None, ii=None):

    if not os.path.isfile(master_raster):
        print("Master raster not created for {}".format(suffix))
        # TODO: More extensive handling of this case.
    else:
        master, meta = load_raster(master_raster)
        class_mask = np.zeros((2, master.shape[1], master.shape[2])) # Just a placeholder
        out = np.zeros((master.shape[2], master.shape[1], num_classes+1))

        # All U-Net specific.
        CHUNK_SIZE = 572
        diff = 92
        stride = 388

        for i in range(0, master.shape[1]-diff, stride):
            for j in range(0, master.shape[2]-diff, stride):
                sub_master = master[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                sub_mask = class_mask[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                sub_master, sub_mask, cut_rows, cut_cols = preprocess_data(sub_master, sub_mask,
                        max_pools, return_cuts=True)
                if channels != 'all':
                    sub_master = sub_master[:, :, :, channels]
                sub_msk = np.ones((1, 388, 388, 5)) # a placeholder
                if sub_master.shape[1] == 572 and sub_master.shape[2] == 572:
                    preds = model.predict([sub_master, sub_msk])
                    preds_exp = np.exp(preds)
                    preds_softmaxed = preds_exp / np.sum(preds_exp, axis=3, keepdims=True)
                    if np.any(np.isnan(preds)):
                        print("Nan prediction.")
                    preds = preds_softmaxed[0, :, :, :]
                else:
                    continue
                if cut_cols == 0 and cut_rows == 0:
                    out[j+diff:j+CHUNK_SIZE-diff, i+diff:i+CHUNK_SIZE-diff, :] = preds
                elif cut_cols == 0 and cut_rows != 0:
                    ofs = master.shape[1]-cut_rows
                    out[j+diff:j+CHUNK_SIZE-diff, i+diff:ofs-diff, :] = preds
                elif cut_cols != 0 and cut_rows == 0:
                    ofs = master.shape[2]-cut_cols
                    out[j+diff:ofs-diff, i+diff:i+CHUNK_SIZE-diff, :] = preds
                elif cut_cols != 0 and cut_rows != 0:
                    ofs_col = master.shape[2]-cut_cols
                    ofs_row = master.shape[1]-cut_rows
                    out[j+diff:ofs_col-diff, i+diff:ofs_row-diff, :] = preds
                else:
                    print("whatcha got goin on here?")

            stdout.write("N eval: {}. Percent done: {:.4f}\r".format(ii, i / master.shape[1]))

    out = np.swapaxes(out, 0, 2)
    out = out.astype(np.float32)
    if outfile:
        save_raster(out, outfile, meta)
    return out


def evaluate_images(image_directory, model, include_string, max_pools, exclude_string, prefix,
        save_dir, channels):
    ii = 0
    for f in glob(os.path.join(image_directory, "*.tif")):
        if exclude_string not in f and include_string in f:
            print(f)
            out = os.path.basename(f)
            os.path.split(out)[1]
            out = out[out.find("_"):]
            out = os.path.splitext(out)[0]
            out = prefix + out + ".tif"
            out = os.path.join(save_dir, out)
            ii += 1
            evaluate_image_unet(f, model, max_pools=max_pools, channels=channels,
                    outfile=out, ii=ii)


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


def train_model(training_directory, model, steps_per_epoch, valid_steps, max_pools, box_size=0,
        epochs=3, random_sample=False, threshold=0.9, w0=50, sigma=10, channels='all', 
        restore=False, learning_rate=1e-3, num_classes=4):
    ''' This function assumes that train/test data are
    subdirectories of training_directory, with
    the names train/test.'''
    if channels == 'all':
        channel_depth = 51
    else:
        channel_depth = channels.shape[0]
    shp = (572, 572, channel_depth)
    model = model(shp, num_classes+1) # + 1 for border class
    model.compile(
             loss=weighted_loss,
             optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
             metrics=[m_acc]
             )
    graph_path = os.path.join('graphs/', str(int(time.time())))
    os.mkdir(graph_path)
    tb = TensorBoard(log_dir=graph_path)
    train = os.path.join(training_directory, 'train')
    test = os.path.join(training_directory, 'test')
    class_weight = {0:25.923, 1:1.0, 2:2.79, 3:61.128}
    train_generator = generate_training_data(train, max_pools, sample_random=random_sample,
            box_size=box_size, batch_size=4, w0=w0, sigma=sigma,
            class_weights=class_weight, channels=channels)
    test_generator = generate_training_data(test, max_pools, sample_random=random_sample,
            train=False, box_size=box_size, batch_size=4, 
            class_weights=class_weight, channels=channels)
    model.fit_generator(train_generator, 
            steps_per_epoch=steps_per_epoch, 
            validation_data=test_generator,
            validation_steps=valid_steps,
            epochs=epochs,
            verbose=1,
            callbacks=[tb, tf.keras.callbacks.TerminateOnNaN()],
            use_multiprocessing=True)

    return model, graph_path


def save_model_info(outfile, args):
    template = '{}={}|'
    with open(outfile, 'a') as f:
        for key in args:
            f.write(template.format(key, args[key]))
        f.write("\n-------------------\n")
    print("wrote run info to {}".format(outfile))


def gradient_wrt_inputs(model, data):
    layer_output = model.output
    loss = -tf.reduce_mean(layer_output)
    grads = K.gradients(loss, model.input[0])[0]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    weights = np.ones((1, 388, 388, 5))
    results = sess.run(grads, feed_dict={model.input[0]:data, model.input[1]:weights})
    return results


if __name__ == '__main__':

    band_dict = {'B1.TIF':np.arange(0, 2+1), 
    'B10.TIF':np.arange(3, 5+1), 
    'B11.TIF':np.arange(6, 8+1), 
    'B2.TIF':np.arange(9, 11+1), 
    'B3.TIF':np.arange(12, 14+1), 
    'B4.TIF':np.arange(15, 17+1), 
    'B5.TIF':np.arange(18, 20+1), 
    'B6.TIF':np.arange(21, 23+1), 
    'B7.TIF':np.arange(24, 26+1), 
    'B8.TIF':np.arange(27, 29+1), 
    'B9.TIF':np.arange(30, 32+1), 
    'aspect.tif':np.array([33]),
    'elevation_diff.tif':np.array([34]),
    'etr.tif':np.arange(35, 37+1), 
    'pet.tif':np.arange(38, 40+1), 
    'pr.tif':np.arange(41, 43+1), 
    'slope.tif':np.array([44]),
    'tmmn.tif':np.arange(45, 47+1), 
    'tmmx.tif':np.arange(48, 50+1)}

    training_directory = 'training_data/multiclass/'
    info_file = 'run_information.txt'
    max_pools = 0
    model_name = 'unet_border_weights{}.h5'.format(int(time.time()))
    model_name ='unet_gradientwrtinputs.h5'
    model_dir = 'models/'
    info_path = os.path.join(model_dir, info_file)
    model_save_path = os.path.join(model_dir, model_name)

    model_func = weighted_unet_no_transpose_conv

    steps_per_epoch = 334 # 334 the number in the max class.
    valid_steps = 4 # 233
    epochs = 1
    w0 = 0
    sigma = 2
    threshold = 0.9

    train_more = False
    eager = False
    class_weights = False
    learning_rate = 1e-3
    random_sample = False
    augment = False
    exclude = ['etr.tif', 'pet.tif', 'slope.tif', 'tmmn.tif', 'tmmx.tif', 'pr.tif']
    for exx in exclude:
        model_name = 'exclude_{}.h5'.format(exx[:-4])
        model_save_path = os.path.join(model_dir, model_name)
        channels = [band_dict[x] for x in band_dict if exx not in x]
        channels = np.hstack(channels)
        print(model_name)

        raster_name = 'exclude_{}'.format(exx[:-4])
        pr_to_eval = '37_28'
        image_directory = 'master_rasters/test/'

        param_dict = {'model_name':model_name, 'epochs':epochs, 'steps_per_epoch':steps_per_epoch,
                'raster_name':raster_name, 'learning_rate':learning_rate, 'eager':eager,
                'class_weights':class_weights, 'augmented':augment, 'random_sample':random_sample,
                'graph_path':None, 'bands':channels, 'w0':w0, 'sigma':sigma}

        evaluating = True
        if not os.path.isfile(model_save_path):
            print("Training new model")
            model, graph_path = train_model(training_directory, model_func,
                    steps_per_epoch=steps_per_epoch, valid_steps=valid_steps,
                    max_pools=max_pools, epochs=epochs, random_sample=random_sample,
                    learning_rate=learning_rate, channels=channels, w0=w0, sigma=sigma,
                    threshold=threshold)
            evaluating = False
            model.save(model_save_path)
        else:
            model = tf.keras.models.load_model(model_save_path,
                    custom_objects={'weighted_loss':weighted_loss})
            model.compile(
                     loss=weighted_loss,
                     optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                     )
            if train_more:
                model, graph_path = train_model(training_directory, model, steps_per_epoch=steps_per_epoch,
                        valid_steps=valid_steps, channel_depth=channel_depth, random_sample=random_sample,
                        max_pools=max_pools, epochs=epochs, restore=True)
                model_name = 'unet_random_sample100.h5'
                model.save(os.path.join(model_dir, model_name))

        if not evaluating or train_more:
            param_dict['graph_path'] = graph_path
            save_model_info(info_path, param_dict)
        # s = '1553014193.4813933'
        # f = 'training_data/multiclass/train/class_2_data/{}.pkl'.format(s)
        # with open(f, 'rb') as f:
        #     data = pload(f)
        # data = np.expand_dims(data['data'], axis=0)
        # data = np.swapaxes(data, 1, 3)
        # gradient_wrt_inputs(model, data)
        evaluate_images(image_directory, model, include_string=pr_to_eval, 
             exclude_string="class", channels=channels, max_pools=max_pools, prefix=raster_name, save_dir='compare_model_outputs/blurry/') 
        #clip_rasters('compare_model_outputs/blurry/', pr_to_eval)
