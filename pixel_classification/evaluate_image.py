import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sys import stdout
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from rasterio import open as rasopen
from glob import glob
import numpy.ma as ma
import tensorflow as tf
from tensorflow.keras.models import load_model

def get_weights(path):
    model = tf.keras.models.load_model(path)
    return model.weights

def sub_img_list(im, kernel_size):
    ofs = kernel_size // 2
    ls = []
    for i in range(kernel_size, im.shape[1]):
        sub_imgs = np.zeros((im.shape[2]-kernel_size, 36, kernel_size, kernel_size))
        k = 0
        for j in range(kernel_size, im.shape[2]):
            sub_img = im[:, i-kernel_size:i, j-kernel_size:j] 
            sub_imgs[k, :, :, :] = sub_img
            k += 1

        ls.append(sub_imgs)
        if i % 2 == 0:
            yield ls

class Result:

    def __init__(self, data, idx):
        self.data = data
        self.idx = idx

def write_raster(data, name, raster_geo):

    raster_geo['dtype'] = data.dtype
    raster_geo['count'] = 1
    with rasopen(name, 'w', **raster_geo) as dst:
        dst.write(data)
    return None

def split_image(image, kernel_size):
    num_rows = image.shape[1] // os.cpu_count()
    leftover = image.shape[1] % os.cpu_count()
    ids = []
    arrs = []
    j = 0
    for idx, i in enumerate(range(kernel_size, image.shape[1], num_rows)):
        arrs.append(image[:, i-kernel_size:i+num_rows+kernel_size:, :])
        ids.append(j)
        j += 1

    arrs.append(image[ :, image.shape[1]-leftover-kernel_size:, :])
    ids.append(j)
    return arrs, ids 

def pool_job(path, image, ids):
    model = Network(path)
    while True:
        eval_image(image, model, ids)
        queue.put(os.getpid())

def is_target(f, targets):

    for ff in targets:
        if ff in f:
            return True
    return False

def get_prev_mask(target):

    for f in glob('evaluated_images/' + "*.npy"):
        if target in f and 'running' in f:
            return f
    return None

def eval_image(im, msk, idd):
    model_path = 'models/model_kernel_41'
    model = load_model(model_path)
    kernel_size = 41
    mask = np.zeros((im.shape[1], im.shape[2]))
    if msk is not None:
        msk = np.load(msk)
        mask[:msk.shape[0], :] = msk 
        begin = msk.shape[0]
        del msk
    else:
        begin = kernel_size 
    ofs = kernel_size // 2
    for i in range(begin, im.shape[1]):
        sub_imgs = np.zeros((im.shape[2]-kernel_size, 36, kernel_size, kernel_size))
        k = 0
        for j in range(kernel_size, im.shape[2]):
            sub_img = im[:, i-kernel_size:i, j-kernel_size:j] 
            sub_imgs[k, :, :, :] = sub_img
            k += 1

        result = model.predict(sub_imgs)
        result = np.argmax(result, axis=1)
        mask[i-ofs, kernel_size - ofs: -(kernel_size-ofs-1)] = result
        if i % 100 == 0:
            np.save("evaluated_images/{}_running_eval".format(idd), mask[:i, :])
            stdout.write("\r{:.5f}".format(float(i)/im.shape[1]))

    np.save("evaluated_images/eval_{}".format(idd), mask)
    return Result(mask, idd)


if __name__ == '__main__':

    path = 'models/model_kernel_41'
    targets = ['38_27_2013', '40_26_2013', '40_27_2013', '39_27_2013', 
            '39_26_2013']
    i = 0 
    kernel_size = 41
    for f in glob("master_rasters/to_eval/" + "*.tif"):
            stdout.write("\rEvaluating image {}\n".format(f))
            with rasopen(f, 'r') as src:
                raster_geo = src.meta.copy()
                im = src.read()
            eval_image(im, None, os.path.basename(f))



