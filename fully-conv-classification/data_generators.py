import numpy as np
import numpy.ma as ma
import os
import time
import pickle
import warnings
import pdb
import matplotlib.pyplot as plt

from glob import glob
from random import sample, shuffle, choice
from scipy.ndimage.morphology import distance_transform_edt
from rasterio import open as rasopen
from rasterio.errors import RasterioIOError
from skimage import transform
from sat_image.warped_vrt import warp_single_image
from tensorflow.keras.utils import Sequence
from multiprocessing import Pool
from collections import defaultdict
from sys import getsizeof

from data_utils import load_raster, paths_map_multiple_scenes, stack_rasters, stack_rasters_multiprocess, download_from_pr
from shapefile_utils import get_shapefile_path_row, mask_raster_to_shapefile, filter_shapefile_overlapping, mask_raster_to_features


class RandomMajorityUndersamplingSequence(Sequence):

    def __init__(self, batch_size, data_directory, training=True):

        self.training = training
        self.data_directory = data_directory
        self.batch_size = batch_size
        self._get_files()
        self.n_files = len(self.file_list)
        shuffle(self.file_list)
        self.idx = 0


    def _get_files(self):

        self.class_directories = os.listdir(self.data_directory)
        self.n_classes = len(self.class_directories)
        self.files = [glob(os.path.join(self.data_directory, d, "*.pkl")) for d in
                self.class_directories]
        self.n_minority = min([len(f) for f in self.files])
        self.file_list = []
        if self.training:
            self.file_list.extend(sample(self.files[0], self.n_minority))
            self.file_list.extend(sample(self.files[1], self.n_minority))
        else:
            self.file_list.extend(self.files[0])
            self.file_list.extend(self.files[1])
        shuffle(self.file_list)


    def __len__(self):
        return int(np.ceil(self.n_files / self.batch_size))


    def on_epoch_end(self):
        self.file_list = []
        self.file_list.extend(sample(self.files[0], self.n_minority))
        self.file_list.extend(sample(self.files[1], self.n_minority))
        shuffle(self.file_list)
        self.n_files = len(self.file_list)


    def __getitem__(self, idx):
        batch = self.file_list[idx * self.batch_size:(idx + 1)*self.batch_size]
        data_tiles = [self._from_pickle(x) for x in batch]
        self.batch=batch
        if self.n_classes == 2:
            processed = self._binary_labels_and_features(data_tiles)
        else:
            processed = self._labels_and_features(data_tiles)

        batch_y = processed[1]
        batch_x = processed[0]
        return batch_x, batch_y
    

    def _from_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data


    def _labels_and_features(self, data_tiles):
        features = []
        one_hots = []
        for tile in data_tiles:
            data = tile['data']
            one_hot = tile['one_hot'].astype(np.int)
            one_hot[0, 0, :] = 0
            class_code = tile['class_code']
            data, one_hot = _augment_data(data, one_hot)
            features.append(data)
            one_hots.append(one_hot)
        return [np.asarray(features)], [np.asarray(one_hots)]


    def _binary_labels_and_features(self, data_tiles):
        features = []
        one_hots = []
        bad_shape = False
        for cnt, tile in enumerate(data_tiles):
            data = tile['data']
            one_hot = tile['one_hot'].astype(np.int)
            binary_one_hot = np.ones((one_hot.shape[0], one_hot.shape[1])).astype(np.int)*-1 
            for i in range(one_hot.shape[2]):
                if i == 1:
                    binary_one_hot[:, :][one_hot[:, :, i] == 1] = 1
                else:
                    binary_one_hot[:, :][one_hot[:, :, i] == 1] = 0
            neg_examples = np.where(binary_one_hot == 0)
            n_neg = len(neg_examples[0])
            n_pos = len(np.where(binary_one_hot == 1)[0])
            if n_neg != 0:
                xs = np.random.choice(neg_examples[0], n_neg - n_pos, replace=False)
                ys = np.random.choice(neg_examples[1], n_neg - n_pos, replace=False)
                binary_one_hot[xs, ys] = -1
            if self.training:
                data, binary_one_hot = _augment_data(data, binary_one_hot, binary=True)
            binary_one_hot = np.expand_dims(binary_one_hot, 2)
            features.append(data)
            one_hots.append(binary_one_hot)
        return [np.asarray(features)], [np.asarray(one_hots)]


class BinaryDataSequence(Sequence):

    def __init__(self, batch_size, file_list, training=True, balance_pixels=False):
        # this requires a file list of training data.
        self.training = training
        self.balance_pixels = balance_pixels
        self.batch_size = batch_size
        self.file_list = file_list
        self.n_files = len(self.file_list)
        shuffle(self.file_list)
        self.idx = 0


    def __len__(self):
        return int(np.ceil(self.n_files / self.batch_size))


    def on_epoch_end(self):
        shuffle(self.file_list)


    def __getitem__(self, idx):
        batch = self.file_list[idx * self.batch_size:(idx + 1)*self.batch_size]
        data_tiles = [self._from_pickle(x) for x in batch]
        processed = self._binary_labels_and_features(data_tiles)
        batch_y = processed[1]
        batch_x = processed[0]
        return batch_x, batch_y
    

    def _from_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data


    def _apply_weights(self, one_hot):
        for i in range(self.n_classes):
            one_hot[:, :, i] *= self.class_weights[i]


    def _binary_labels_and_features(self, data_tiles):
        features = []
        one_hots = []
        for tile in data_tiles:
            data = tile['data']
            one_hot = tile['one_hot'].astype(np.int)
            binary_one_hot = np.ones((one_hot.shape[0], one_hot.shape[1])).astype(np.int)*-1 
            for i in range(one_hot.shape[2]):
                if i == 1:
                    binary_one_hot[:, :][one_hot[:, :, i] == 1] = 1
                else:
                    binary_one_hot[:, :][one_hot[:, :, i] == 1] = 0
            if self.training:
                data, binary_one_hot = _augment_data(data, binary_one_hot, binary=True)
            if self.training and self.balance_pixels:
                neg_examples = np.where(binary_one_hot == 0)
                n_neg = len(neg_examples[0])
                n_pos = len(np.where(binary_one_hot == 1)[0])
                if n_neg > n_pos:
                    idx = np.random.choice(np.arange(n_neg), n_neg - n_pos, replace=False)
                    xs = neg_examples[0][idx]
                    ys = neg_examples[1][idx]
                    binary_one_hot[xs, ys] = -1
            binary_one_hot = np.expand_dims(binary_one_hot, 2)
            features.append(data)
            one_hots.append(binary_one_hot)
        return [np.asarray(features)], [np.asarray(one_hots)]


def _flip_lr(feature_tile, one_hot, binary=False):
    for i in range(feature_tile.shape[2]):
        feature_tile[:, :, i] = np.fliplr(feature_tile[:, :, i])
    if binary:
        one_hot = np.fliplr(one_hot)
        return feature_tile, one_hot
    for i in range(one_hot.shape[2]):
        one_hot[:, :, i] = np.fliplr(one_hot[:, :, i])
    return feature_tile, one_hot


def _flip_ud(feature_tile, one_hot, binary=False):

    for i in range(feature_tile.shape[2]):
        feature_tile[:, :, i] = np.flipud(feature_tile[:, :, i])
    if binary:
        one_hot = np.flipud(one_hot)
        return feature_tile, one_hot
    for i in range(one_hot.shape[2]):
        one_hot[:, :, i] = np.flipud(one_hot[:, :, i])
    return feature_tile, one_hot


def _flip_lr_ud(feature_tile, one_hot, binary=False):
    feature_tile, one_hot = _flip_lr(feature_tile, one_hot, binary)
    feature_tile, one_hot = _flip_ud(feature_tile, one_hot, binary)
    return feature_tile, one_hot


def _do_nothing(feature_tile, one_hot, binary):
    return feature_tile, one_hot


def _augment_data(feature_tile, one_hot, binary=False):
    ''' Applies rotation | lr | ud | lr_ud | flipping, or doesn't. '''
    possible_augments = [_flip_ud, _flip_lr, _flip_lr_ud, _do_nothing]
    return choice(possible_augments)(feature_tile, one_hot, binary)


if __name__ == '__main__':
    pass
