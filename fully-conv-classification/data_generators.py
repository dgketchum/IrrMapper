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
from skimage.morphology import erosion
from sat_image.warped_vrt import warp_single_image
from tensorflow.keras.utils import Sequence
from multiprocessing import Pool
from collections import defaultdict
from sys import getsizeof, exit

from data_utils import load_raster, paths_map_multiple_scenes, stack_rasters, stack_rasters_multiprocess, download_from_pr
from shapefile_utils import get_shapefile_path_row, mask_raster_to_shapefile, filter_shapefile_overlapping, mask_raster_to_features


class SatDataGenerator(Sequence):

    def __init__(self, batch_size, n_classes, balance_pixels_per_batch=False, training=True,
            apply_irrigated_weights=False):

        self.batch_size = batch_size
        self.n_classes = n_classes
        self.training = training
        self.balance_pixels_per_batch = balance_pixels_per_batch
        self.apply_irrigated_weights = apply_irrigated_weights

    def _get_files(self):
        # Required override.
        raise NotImplementedError


    def on_epoch_end(self):
        raise NotImplementedError


    def __len__(self):
        raise NotImplementedError


    def __getitem__(self, idx):
        raise NotImplementedError


    def _from_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data


    def _labels_and_features(self, data_tiles):
        features = []
        one_hots = []
        if self.balance_pixels_per_batch:
            min_count = self._count_pixels(data_tiles)
        
        for tile in data_tiles:
            data = tile['data']
            one_hot = tile['one_hot'].astype(np.int)
            if self.apply_irrigated_weights:
                one_hot[:, :, 0] *= 50
            class_code = tile['class_code']
            if self.balance_pixels_per_batch:
                one_hot = self._balance_pixels(one_hot, min_count)
            if self.training:
                data, one_hot = _augment_data(data, one_hot)
            features.append(data)
            one_hots.append(one_hot)

        return [np.asarray(features)], [np.asarray(one_hots)]


    def _balance_pixels(self, one_hot, min_count, binary=False):

        if binary:
            ys, xs = np.where(one_hot[:, :] == 1)
            if len(ys):
                ys = np.random.choice(ys, size=int(len(ys)-min_count), replace=False)
                xs = np.random.choice(xs, size=int(len(xs)-min_count), replace=False)
                one_hot[ys, xs, i] = -1
            ys, xs = np.where(one_hot[:, :] == 0)
            if len(ys):
                ys = np.random.choice(ys, size=int(len(ys)-min_count), replace=False)
                xs = np.random.choice(xs, size=int(len(xs)-min_count), replace=False)
                one_hot[ys, xs, i] = -1

        else:
            for i in range(one_hot.shape[2]):
                ys, xs = np.where(one_hot[:, :, i] == 1)
                if len(ys) == min_count:
                    continue
                if len(ys):
                    n_to_choose = int(len(ys) - min_count) # 0 out all but min_count pixels
                    ys = np.random.choice(ys, size=n_to_choose, replace=False)
                    xs = np.random.choice(xs, size=n_to_choose, replace=False)
                    one_hot[ys, xs, i] = 0
            
        return one_hot


    def _count_pixels(self, data_tiles):
        pixel_counts = np.ones((self.n_classes))*np.inf
        for tile in data_tiles:
            data = tile['data']
            one_hot = tile['one_hot'].astype(np.int)
            nodata_mask = np.sum(one_hot, axis=2) 
            argmaxed = np.argmax(one_hot, axis=2) 
            argmaxed[nodata_mask == 0] = -1
            unique, counts = np.unique(argmaxed, return_counts=True) 
            unique = unique[1:]
            counts = counts[1:] 
            for val, count in zip(unique, counts):
                if count < pixel_counts[val]:
                    pixel_counts[val] = count
        return np.min(pixel_counts)
    

    def _binary_labels_and_features(self, data_tiles):
        features = []
        one_hots = []

        if self.balance_pixels_per_batch:
            min_count = self._count_pixels(data_tiles)
        
        for cnt, tile in enumerate(data_tiles):
            data = tile['data']
            one_hot = tile['one_hot'].astype(np.int)
            binary_one_hot = np.ones((one_hot.shape[0], one_hot.shape[1])).astype(np.int)*-1 
            for i in range(one_hot.shape[2]):
                if i == 1:
                    binary_one_hot[:, :][one_hot[:, :, i] == 1] = 1
                else:
                    binary_one_hot[:, :][one_hot[:, :, i] == 1] = 0
            
            if self.balance_pixels_per_batch:
                one_hot = self._balance_pixels(one_hot, min_count, binary=True)

            if self.training:
                data, binary_one_hot = _augment_data(data, binary_one_hot, binary=True)
            binary_one_hot = np.expand_dims(binary_one_hot, 2)
            features.append(data)
            one_hots.append(binary_one_hot)
        return [np.asarray(features)], [np.asarray(one_hots)]


class DataGenerator(SatDataGenerator):
    '''
    Feeds examples into the network in order
    sorted by class_label. This is a form of random majority 
    undersampling.

    I want the following functionality:
       Easily switch between binary/multiclass classification

       Can focus on examples from one class (list of target_classes) 
       Can apply arbitary morphological operations to the input labels

       Able to feed in examples without any preprocessing (unbalanced)
       Able to feed in examples that are balanced, but in a random order
       Able to feed in examples that are balanced and in a definite order (queue of files)
       Able to feed in batches that are balanced on a pixel count level.

    '''
    def __init__(self, data_directory, batch_size, n_classes=None, training=True,
            target_classes=None, balance=False, balance_examples_per_batch=False,
            balance_pixels_per_batch=False, apply_irrigated_weights=False,
            steps_per_epoch=None):
        # Assert that all three can't be true
        super().__init__(batch_size, n_classes, balance_pixels_per_batch, training)
        self.data_directory = data_directory
        self.balance = balance
        self.balance_examples_per_batch = balance_examples_per_batch
        self.target_classes = target_classes
        self.steps_per_epoch = steps_per_epoch
        self._get_files()


    def _check_if_directory_is_in_targets(self, directory):
        ''' Assumes directory is at the top of the 
            directory hierarchy '''
        if self.target_classes is None:
            return True
        if not isinstance(self.target_classes, list):
            if isinstance(self.target_classes, int):
                self.target_classes = [self.target_classes]
            else:
                raise ValueError("target_classes must be one of int, list")
        for target in self.target_classes:
            if str(target) in directory:
                return True
        return False


    def _get_files(self):
        dirs = os.listdir(self.data_directory)
        if self.n_classes is None:
            self.n_classes = len(dirs)
        for d in os.listdir(self.data_directory):
            if not os.path.isdir(os.path.join(self.data_directory, d)):
                raise ValueError("Non-directory object exists in data_directory")
        dirs = [os.path.join(self.data_directory, d) for d in dirs \
                if self._check_if_directory_is_in_targets(d)]
        self.dirs = dirs
        if not self.balance and not self.balance_examples_per_batch:
            self.n_files = self._unbalanced_file_list(dirs, first=True)
            self._on_epoch_end = self._unbalanced_file_list
            # all training examples, randomly selected
            # number of files in an epoch is the sum of the files
            # for each class
            return
        elif self.balance:
            self.n_files = self._balanced_file_list(dirs, first=True)
            self._on_epoch_end = self._balanced_file_list
            # balanced file list with random selection
            # i.e. the number of files in an epoch
            # is n classes * min number of training examples for any class
            return
        if self.balance_examples_per_batch:
            self.n_files = self._balanced_queue(dirs, first=True)
            self._on_epoch_end = self._balanced_queue
            # all training examples, fed to the network in sequential order 
            # i.e. 1, 2, 3, 4, 1, 2, 3, 4
            # the number of files in an epoch is 
            # n classes * min number of training examples for any class
            return

    def __len__(self):
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch
        return int(np.ceil(self.n_files / self.batch_size))


    def on_epoch_end(self):
        # Recreates the file list
        self._on_epoch_end(self.dirs, first=False)


    def __getitem__(self, idx):
        # model.fit_generator does not pull batches in order.
        batch = self.files[idx * self.batch_size:(idx + 1)*self.batch_size]
        data_tiles = [self._from_pickle(x) for x in batch]
        self.batch = batch
        if self.n_classes == 2:
            batch_x, batch_y = self._binary_labels_and_features(data_tiles)
        else:
            batch_x, batch_y = self._labels_and_features(data_tiles)

        return batch_x, batch_y
    


    def _unbalanced_file_list(self, dirs, first):
        if first:
            self.files = []
            for d in dirs:
                self.files.extend(glob(os.path.join(d, "*pkl")))
            return len(self.files)
        else:
            shuffle(self.files)
    
    
    def _balanced_file_list(self, dirs, first):
        if first:
            self.file_dict = {}
            self.n_minority = np.inf
            for d in dirs:
                files = glob(os.path.join(d, "*pkl"))
                shuffle(files)
                self.file_dict[d] = files
                if len(files) < self.n_minority:
                    self.n_minority = len(files)
            self.files = []
            for key in self.file_dict:
                self.files.extend(sample(self.file_dict[key], self.n_minority))
            return len(self.files)
        else:
            self.files = []
            for key in self.file_dict:
                self.files.extend(sample(self.file_dict[key], self.n_minority))
            shuffle(self.files)

    
    def _balanced_queue(self, dirs, first):
        self.file_dict = {}
        self.n_minority = np.inf
        for d in dirs:
            files = glob(os.path.join(d, "*pkl"))
            shuffle(files)
            self.file_dict[d] = files
            if len(files) < self.n_minority:
                self.n_minority = len(files)
        if not first:
            for key in self.file_dict:
                shuffle(self.file_dict[key])
        self.files = []
        to_empty = self.file_dict.copy()
        while True:
            try:
                for key in sorted(to_empty):
                    # python3 supports ordered iteration over dict keys.
                    # this may not work perfectly when
                    # fitting a generator with use_multiprocessing == True
                    self.files.append(to_empty[key].pop())
            except IndexError as e:
                break
        return len(self.files)


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
