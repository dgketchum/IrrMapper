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
from tensorflow.keras.callbacks import (TensorBoard, ModelCheckpoint, LearningRateScheduler,
        ReduceLROnPlateau)
from rasterio import open as rasopen
from rasterio.mask import mask
from shapely.geometry import shape
from pickle import load as pload
from fiona import open as fopen
from data_generators import generate_training_data, load_raster
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
    # Weight map:
    out = -tf.reduce_sum(target*output, len(output.get_shape())-1)
    #mask = tf.not_equal(out, 0)#tf.boolean_mask(out, mask)
    return out

def weighted_focal_loss(target, output, gamma=1):
    exp = tf.exp(output)
    pt = tf.pow(1-exp, gamma)
    out = -tf.reduce_sum(target*output, len(output.get_shape())-1)
    mask = tf.not_equal(out, 0)
    pt_ce = tf.multiply(pt, output)
    out = -tf.reduce_sum(pt_ce*target, len(output.get_shape()) -1)
    return tf.boolean_mask(out, mask)
    

def acc(y_true, y_pred):
    y_pred_sum = tf.reduce_sum(y_pred, axis=3) 
    mask = tf.not_equal(y_pred_sum, 0)
    y_arg = tf.argmax(y_pred, axis=-1)
    y_t_arg = tf.argmax(y_true, axis=-1)
    y_arg_mask = tf.boolean_mask(y_arg, mask)
    y_t_arg_mask = tf.boolean_mask(y_t_arg, mask)
    return K.mean(K.equal(y_t_arg_mask, y_arg_mask))


def evaluate_image(master_raster, model, max_pools, channels='all', num_classes=4,
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

            stdout.write("N eval: {}. Percent done: {:.2f}\r".format(ii, i / master.shape[1]))

    out = np.swapaxes(out, 0, 2)
    out = out.astype(np.float32)
    if outfile:
        save_raster(out, outfile, meta, count=5)
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


def lr_schedule(epoch, lr):
    return 0.01*np.exp(-epoch/1000)


def train_model(training_directory, model, steps_per_epoch, valid_steps, max_pools, box_size=0,
        epochs=3, random_sample=False, threshold=None, sigma=None, w0=None, channels='all', train_more=False, raster_name=None, learning_rate=1e-3, num_classes=5):
    ''' This function assumes that train/test data are
    subdirectories of training_directory, with
    the names train/test.'''
    if channels == 'all':
        channel_depth = 51
    else:
        channel_depth = channels.shape[0]
    shp = (572, 572, channel_depth)
    weight_shape = (388, 388, num_classes)
    if not train_more:
        model = model(shp, weight_shape, num_classes, base_exp=5) 
        model.compile(
                 loss=weighted_focal_loss,
                 optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                 metrics=[acc]
                 )
        model.summary()
    graph_path = os.path.join('graphs/', str(int(time.time())))
    os.mkdir(graph_path)
    tb = TensorBoard(log_dir=graph_path)
    ckpt_path = os.path.join(graph_path, raster_name+"_{epoch:02d}-{val_acc:.2f}.hdf5")
    scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    mdlcheck = ModelCheckpoint(ckpt_path, monitor='val_acc', save_best_only=True,
            mode='max', verbose=1)
    train = os.path.join(training_directory, 'train')
    test = os.path.join(training_directory, 'test')
    class_weight = {0:28.101, 1:1.0, 2:2.9614, 3:103.8927} #for no buffer
    #class_weight = {0:1, 1:1.0, 2:1, 3:1} #for no buffer

    train_generator = generate_training_data(train, max_pools, sample_random=random_sample,
            box_size=box_size, class_weights=class_weight, channels=channels, threshold=threshold,
            sigma=sigma, w0=w0)
    test_generator = generate_training_data(test, max_pools, sample_random=False,
            box_size=box_size, batch_size=4, w0=w0, threshold=threshold,
            sigma=sigma, class_weights=class_weight, channels=channels)

    model.fit_generator(train_generator, 
            steps_per_epoch=steps_per_epoch, 
            epochs=epochs,
            verbose=1,
            validation_data=test_generator,
            validation_steps=valid_steps,
            callbacks=[tb, scheduler, mdlcheck, tf.keras.callbacks.TerminateOnNaN()],
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
    # s = '1553014193.4813933'
    # f = 'training_data/multiclass/train/class_2_data/{}.pkl'.format(s)
    # with open(f, 'rb') as f:
    #     data = pload(f)
    # data = np.expand_dims(data['data'], axis=0)
    # data = np.swapaxes(data, 1, 3)
    # gradient_wrt_inputs(model, data)
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

    training_directory = 'training_data/'
    info_file = 'run_information.txt'
    max_pools = 0
    model_dir = 'models/'
    info_path = os.path.join(model_dir, info_file)
    model_func = weighted_unet_no_transpose_conv
    steps_per_epoch = 100
    valid_steps = 20
    epochs = 320
    w0 = 10
    sigma = 5 
    threshold = 0.7*w0
    train_iter = 1
    train_more = False
    eager = False
    class_weights = True
    learning_rate = 1e-3
    random_sample = False
    augment = False
    channels = 'all' 
    raster_name = '2_w0:{}-th:{}-sigma:{}-lr:{}'.format(w0, threshold, sigma, learning_rate)
    model_name = '2_w0:{}-th:{}-sigma:{}-lr:{}'.format(w0, threshold, sigma, learning_rate)
    raster_name = 'unit_irr_weights_normal_loss_w0:{}-th:{}-sigma:{}-lr:{}'.format(w0, threshold, sigma, learning_rate)
    model_name = 'unit_1border_weights_irr_weights_normal_loss_w0_weight:{}-th:{}-sigma:{}-lr:{}'.format(w0, threshold, sigma, learning_rate)
    # Next try raw weights from weight map.
    model_save_path = os.path.join(model_dir, model_name)
    pr_to_eval = '37_28'
    if pr_to_eval == '39_27':
        image_directory = '/home/thomas/share/master_rasters/train/'
    else:
        image_directory = '/home/thomas/share/master_rasters/test/'

    param_dict = {'model_name':model_name, 'epochs':epochs, 'steps_per_epoch':steps_per_epoch,
            'raster_name':raster_name, 'learning_rate':learning_rate, 'eager':eager,
            'class_weights':class_weights, 'augmented':augment, 'random_sample':random_sample,
            'graph_path':None, 'bands':channels, 'w0':w0, 'sigma':sigma}

    evaluating = True
    num_classes = 5
    model_save_path = 'models/modelsunit_1border_weights_irr_weights_normal_loss_w0_weight:10-th:7.0-sigma:5-lr:0.001step_2850'
    if not os.path.isfile(model_save_path):
        print("Training new model")
        shp = (572, 572, 51)
        weight_shape = (388, 388, num_classes)
        model = weighted_unet_no_transpose_conv(shp, weight_shape, num_classes, base_exp=5) 
        model.compile(
                 loss=weighted_loss,
                 optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                 metrics=[acc]
                 )
        graph_path = os.path.join('graphs/', str(int(time.time())))
        os.mkdir(graph_path)
        tb = TensorBoard(log_dir=graph_path)
        ckpt_path = os.path.join(graph_path, raster_name+"_{epoch:02d}-{val_acc:.2f}.hdf5")
        scheduler = LearningRateScheduler(lr_schedule, verbose=1)
        mdlcheck = ModelCheckpoint(ckpt_path, monitor='val_acc', save_best_only=True,
                mode='max', verbose=1)
        train = os.path.join(training_directory, 'train')
        test = os.path.join(training_directory, 'test')
        class_weight = {0:28.101, 1:1.0, 2:2.9614, 3:103.8927} #for no buffer
        #class_weight = {0:1, 1:1.0, 2:1, 3:1} #for no buffer

        train_generator = generate_training_data(train, max_pools, sample_random=False,
                class_weights=class_weight, channels=channels, threshold=threshold,
                sigma=sigma, w0=w0)
        i = 0
        k = 0
        train_iter = 150
        for data, labels in train_generator:
            out = model.train_on_batch(x=data,
                    y=labels)
            # Loss, accuracy?
            print(out)
            if i > train_iter:
                model.save('models'+model_name+'step_{}'.format((k+1)*train_iter))
                evaluate_images(image_directory, model, include_string=pr_to_eval,
                        exclude_string="class", channels=channels, max_pools=max_pools,
                        prefix=raster_name+'step_{}'.format((k+1)*train_iter),
                        save_dir='compare_model_outputs/during-the-day/') 
                k+=1
                i = 0
            i += 1
        
    raster_name='final_eval'
    model = tf.keras.models.load_model(model_save_path,
            custom_objects={'weighted_loss':weighted_loss})
    evaluate_images(image_directory, model, include_string=pr_to_eval, exclude_string="class", channels=channels, max_pools=max_pools, prefix=raster_name, save_dir='compare_model_outputs/systematic/') 
