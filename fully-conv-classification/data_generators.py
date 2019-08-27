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

from runspec import mask_rasters
from data_utils import load_raster, paths_map_multiple_scenes, stack_rasters, stack_rasters_multiprocess, download_from_pr
from shapefile_utils import get_shapefile_path_row, mask_raster_to_shapefile, filter_shapefile_overlapping, mask_raster_to_features


class SatDataSequence(Sequence):

    def __init__(self, data_directory, batch_size, class_weights=None,
            balance=True, single_class_per_tile=True, n_classes=5, classes_to_augment=None):
        self.data_directory = data_directory
        self.class_weights = class_weights
        if self.class_weights is None:
            dct = {}
            for i in range(n_classes):
                dct[i] = 1
            self.class_weights = dct
        self.n_classes = n_classes
        self.single_class_per_tile = single_class_per_tile
        self.batch_size = batch_size
        self._no_augment = classes_to_augment is None
        self.classes_to_augment = classes_to_augment
        self.balance = balance
        self._get_files()
        self.n_files = len(self.file_list)
        self.idx = 0


    def _get_files(self):
        # Now, get n lists where n is the number of classes (excluding border class).
        # Then, sample from the minority lists until we have 
        # the same number of data tiles from each class, then concatenate
        # all the lists and shuffle. on epoch end, do this process again.
        self.file_dict = {}
        i = 0
        for (dirpath, dirnames, filenames) in os.walk(self.data_directory):
            if dirpath != self.data_directory:
                self.file_dict[i] = [os.path.join(dirpath, x) for x in filenames]
                i += 1
        self.lengths = [len(self.file_dict[k]) for k in self.file_dict]
        self._create_file_list()
        shuffle(self.file_list)



    def _create_file_list(self):
        max_instances = max(self.lengths)
        self.file_list = []
        for class_dir in self.file_dict:
            files = self.file_dict[class_dir]
            self.file_list.extend(files)
            if self.balance:
                if len(files) < max_instances:
                    s = len(files)
                    if len(files) < (max_instances - len(files)):
                        files *= (max_instances // len(files))
                        shuffle(files)
                    additional_files = sample(files, max_instances-s)
                    self.file_list.extend(additional_files)
        shuffle(self.file_list)


    def __len__(self):
        return int(np.ceil(self.n_files / self.batch_size))


    def on_epoch_end(self):
        self._create_file_list()


    def __getitem__(self, idx):
        batch = self.file_list[idx * self.batch_size:(idx + 1)*self.batch_size]
        data_tiles = [self._from_pickle(x) for x in batch]
        processed = self._labels_and_features(data_tiles, self.classes_to_augment)
        batch_x = processed[0]
        batch_y = processed[1]
        return batch_x, batch_y
    

    def _from_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def _apply_weights(self, one_hot):
        for i in range(self.n_classes):
            one_hot[:, :, i] *= self.class_weights[i]


    def _labels_and_features(self, data_tiles, classes_to_augment):
        features = []
        one_hots = []
        for tile in data_tiles:
            data = tile['data']
            one_hot = tile['one_hot'].astype(np.int)
            one_hot[0, 0, :] = 0
            self._apply_weights(one_hot)
            class_code = tile['class_code']
            if not self._no_augment:
                if classes_to_augment[tile['class_code']]:
                    data, one_hot, weights = _augment_data(data, one_hot, weights)
            features.append(data)
            one_hots.append(one_hot)
        return [np.asarray(features)], [np.asarray(one_hots)]


    def _make_weights_labels_and_features(self, data_tiles, classes_to_augment):
        class_weights = self.class_weights
        features = []
        one_hots = []
        weight_list = []
        for tile in data_tiles:
            data = tile['data']
            one_hot = tile['one_hot'].astype(np.int)
            one_hot[0, 0, :] = 0
            weights = tile['weights'].astype(np.int)
            class_code = tile['class_code']
            if not self.single_class_per_tile:
                weights = np.zeros_like(one_hot)
                for cc in range(self.n_classes):
                    for dd in range(self.n_classes):
                        weights[:, :, cc][one_hot[:, :, dd] == 1] = class_weights[dd]

            assert np.all(weights[:, :, 0] == weights[:, :, 1])

            if self.single_class_per_tile:
                for i in range(self.n_classes):
                    weights[:, :, i][one_hot[:, :, class_code] == 1] = class_weights[class_code]

            if class_code == 0 and self.single_class_per_tile:
                for i in range(self.n_classes):
                    weights[:, :, i][one_hot[:, :, self.n_classes-1] == 1] = class_weights[self.n_classes-1]
            if not self._no_augment:
                if classes_to_augment[tile['class_code']]:
                    data, one_hot, weights = _augment_data(data, one_hot, weights)
            features.append(data)
            one_hots.append(one_hot)
            weight_list.append(weights)
        return [np.asarray(features), np.asarray(weight_list)], [np.asarray(one_hots)]


def _flip_lr(feature_tile, one_hot, weights):
    for i in range(feature_tile.shape[2]):
        feature_tile[:, :, i] = np.fliplr(feature_tile[:, :, i])
    for i in range(one_hot.shape[2]):
        one_hot[:, :, i] = np.fliplr(one_hot[:, :, i])
        weights[:, :, i] = np.fliplr(weights[:, :, i])
    return feature_tile, one_hot, weights


def _flip_ud(feature_tile, one_hot, weights):
    for i in range(feature_tile.shape[2]):
        feature_tile[:, :, i] = np.flipud(feature_tile[:, :, i])
    for i in range(one_hot.shape[2]):
        one_hot[:, :, i] = np.flipud(one_hot[:, :, i])
        weights[:, :, i] = np.flipud(weights[:, :, i])
    return feature_tile, one_hot, weights


def _rotate(feature_tile, one_hot, weights):
    # Rotate data.
    rot = np.random.randint(-25, 25)
    for i in range(feature_tile.shape[2]):
        feature_tile[:, :, i] = transform.rotate(feature_tile[:, :, i], rot, cval=0)
    for i in range(one_hot.shape[2]):
        one_hot[:, :, i] = transform.rotate(one_hot[:, :, i], rot, cval=0)
        weights[:, :, i] = transform.rotate(weights[:, :, i], rot, cval=0)
    return feature_tile, one_hot, weights


def _flip_lr_ud(feature_tile, one_hot, weights):
    feature_tile, one_hot, weights = _flip_lr(feature_tile, one_hot, weights)
    feature_tile, one_hot, weights = _flip_ud(feature_tile, one_hot, weights)
    return feature_tile, one_hot, weights


def _do_nothing(feature_tile, one_hot, weights):
    return feature_tile, one_hot, weights


def _augment_data(feature_tile, one_hot, weights):
    ''' Applies rotation | lr | ud | lr_ud | flipping, or doesn't. '''
    possible_augments = [_flip_ud, _flip_lr, _flip_lr_ud, _do_nothing]
    return choice(possible_augments)(feature_tile, one_hot, weights)


if __name__ == '__main__':
    pass
