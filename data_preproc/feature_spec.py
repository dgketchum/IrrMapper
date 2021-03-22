from collections import OrderedDict
import tensorflow as tf

'''
Feature spec for reading/writing tf records
'''

DEFAULT_VALUE = tf.ones([256, 256], dtype=tf.float32) * -1.

features_dict_means = OrderedDict(
    [('0_blue_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('0_green_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('0_red_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('0_nir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('0_swir1_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('0_swir2_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('0_tir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('1_blue_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('1_green_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('1_red_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('1_nir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('1_swir1_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('1_swir2_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('1_tir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('2_blue_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('2_green_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('2_red_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('2_nir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('2_swir1_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('2_swir2_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('2_tir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('3_blue_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('3_green_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('3_red_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('3_nir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('3_swir1_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('3_swir2_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('3_tir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('4_blue_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('4_green_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('4_red_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('4_nir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('4_swir1_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('4_swir2_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('4_tir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('5_blue_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('5_green_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('5_red_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('5_nir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('5_swir1_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('5_swir2_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('5_tir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('6_blue_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('6_green_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('6_red_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('6_nir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('6_swir1_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('6_swir2_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('6_tir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('7_blue_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('7_green_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('7_red_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('7_nir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('7_swir1_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('7_swir2_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('7_tir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('8_blue_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('8_green_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('8_red_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('8_nir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('8_swir1_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('8_swir2_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('8_tir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('9_blue_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('9_green_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('9_red_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('9_nir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('9_swir1_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('9_swir2_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('9_tir_mean', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('elv', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('slp', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('asp', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('lon', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('lat', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('cdl', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('cconf', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('irr', tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=DEFAULT_VALUE))])

features_dict_interp = OrderedDict([
    ('blue_0',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_0',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_0',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_0',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_0',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_0',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_0',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_1',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_1',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_1',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_1',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_1',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_1',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_1',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_2',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_2',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_2',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_2',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_2',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_2',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_2',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_3',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_3',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_3',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_3',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_3',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_3',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_3',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_4',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_4',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_4',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_4',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_4',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_4',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_4',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_5',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_5',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_5',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_5',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_5',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_5',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_5',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_6',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_6',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_6',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_6',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_6',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_6',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_6',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_7',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_7',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_7',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_7',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_7',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_7',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_7',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_8',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_8',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_8',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_8',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_8',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_8',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_8',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_9',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_9',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_9',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_9',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_9',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_9',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_9',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_10',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_10',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_10',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_10',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_10',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_10',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_10',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_11',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_11',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_11',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_11',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_11',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_11',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_11',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('blue_12',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('green_12',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('red_12',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('nir_12',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir1_12',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('swir2_12',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('tir_12',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('lat',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('lon',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('elv',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('slp',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('asp',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('cdl',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('cconf',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE)),
    ('irr',
     tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32,
                           default_value=DEFAULT_VALUE))])


def features_dict(kind='interp'):
    if kind == 'interp':
        print('PROCESSING {} to {} BANDS'.format(kind.upper(), len(features_dict_interp.keys())))
        return features_dict_interp
    elif kind == 'means':
        print('PROCESSING {} to {} BANDS'.format(kind.upper(), len(features_dict_means.keys())))
        return features_dict_means


def bands():
    bands = list(features_dict_interp.keys())
    return bands


def features(kind='interp'):
    if kind == 'interp':
        features = list(features_dict_interp.keys())[:-1]
        for i, k in enumerate(features_dict_interp.keys()):
            print((i, k))
        return features
    elif kind == 'means':
        features = list(features_dict_means.keys())[:-1]
        return features


def enumerated_keys():
    return [(0, 'blue_0'),
            (1, 'green_0'),
            (2, 'red_0'),
            (3, 'nir_0'),
            (4, 'swir1_0'),
            (5, 'swir2_0'),
            (6, 'tir_0'),
            (7, 'blue_1'),
            (8, 'green_1'),
            (9, 'red_1'),
            (10, 'nir_1'),
            (11, 'swir1_1'),
            (12, 'swir2_1'),
            (13, 'tir_1'),
            (14, 'blue_2'),
            (15, 'green_2'),
            (16, 'red_2'),
            (17, 'nir_2'),
            (18, 'swir1_2'),
            (19, 'swir2_2'),
            (20, 'tir_2'),
            (21, 'blue_3'),
            (22, 'green_3'),
            (23, 'red_3'),
            (24, 'nir_3'),
            (25, 'swir1_3'),
            (26, 'swir2_3'),
            (27, 'tir_3'),
            (28, 'blue_4'),
            (29, 'green_4'),
            (30, 'red_4'),
            (31, 'nir_4'),
            (32, 'swir1_4'),
            (33, 'swir2_4'),
            (34, 'tir_4'),
            (35, 'blue_5'),
            (36, 'green_5'),
            (37, 'red_5'),
            (38, 'nir_5'),
            (39, 'swir1_5'),
            (40, 'swir2_5'),
            (41, 'tir_5'),
            (42, 'blue_6'),
            (43, 'green_6'),
            (44, 'red_6'),
            (45, 'nir_6'),
            (46, 'swir1_6'),
            (47, 'swir2_6'),
            (48, 'tir_6'),
            (49, 'blue_7'),
            (50, 'green_7'),
            (51, 'red_7'),
            (52, 'nir_7'),
            (53, 'swir1_7'),
            (54, 'swir2_7'),
            (55, 'tir_7'),
            (56, 'blue_8'),
            (57, 'green_8'),
            (58, 'red_8'),
            (59, 'nir_8'),
            (60, 'swir1_8'),
            (61, 'swir2_8'),
            (62, 'tir_8'),
            (63, 'blue_9'),
            (64, 'green_9'),
            (65, 'red_9'),
            (66, 'nir_9'),
            (67, 'swir1_9'),
            (68, 'swir2_9'),
            (69, 'tir_9'),
            (70, 'blue_10'),
            (71, 'green_10'),
            (72, 'red_10'),
            (73, 'nir_10'),
            (74, 'swir1_10'),
            (75, 'swir2_10'),
            (76, 'tir_10'),
            (77, 'blue_11'),
            (78, 'green_11'),
            (79, 'red_11'),
            (80, 'nir_11'),
            (81, 'swir1_11'),
            (82, 'swir2_11'),
            (83, 'tir_11'),
            (84, 'blue_12'),
            (85, 'green_12'),
            (86, 'red_12'),
            (87, 'nir_12'),
            (88, 'swir1_12'),
            (89, 'swir2_12'),
            (90, 'tir_12'),
            (91, 'lat'),
            (92, 'lon'),
            (93, 'elv'),
            (94, 'slp'),
            (95, 'asp'),
            (96, 'cdl'),
            (97, 'cconf'),
            (98, 'irr')]


if __name__ == '__main__':
    print(len(features()))
    print(len(bands()))
