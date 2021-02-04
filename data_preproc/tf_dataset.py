import os
import tensorflow as tf

from data_preproc import feature_spec

MODE = 'irr'
FEATURES_DICT = feature_spec.features_dict(kind='interp')
FEATURES = feature_spec.features(kind='interp')
step_, length_ = 7, len(FEATURES)
NDVI_INDICES = [(x, y) for x, y in zip(range(2, length_, step_), range(3, length_, step_))]


def make_test_dataset(root, pattern='*gz'):
    training_root = os.path.join(root, pattern)
    datasets = get_dataset(training_root)
    return datasets


def get_dataset(pattern):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
      pattern: A file pattern to match in a Cloud Storage bucket.
    Returns:
      A tf.data.Dataset
    """
    if not isinstance(pattern, list):
        pattern = tf.io.gfile.glob(pattern)
    dataset = tf.data.TFRecordDataset(pattern, compression_type='GZIP',
                                      num_parallel_reads=8)

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    to_tup = to_tuple(add_ndvi=False)
    dataset = dataset.map(to_tup, num_parallel_calls=5)
    return dataset


def parse_tfrecord(example_proto):
    """the parsing function.
    read a serialized example into the structure defined by features_dict.
    args:
      example_proto: a serialized example.
    returns:
      a dictionary of tensors, keyed by feature name.
    """
    parsed = tf.io.parse_single_example(example_proto, FEATURES_DICT)
    return parsed


def to_tuple(add_ndvi):
    """
    Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
      inputs: A dictionary of tensors, keyed by feature name.
    Returns:
      A tuple of (inputs, outputs).
    """

    def to_tup(inputs):
        features_list = [inputs.get(key) for key in FEATURES]
        stacked = tf.stack(features_list, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])  # TC scaled somehow: * 0.0001
        if add_ndvi:
            image_stack = add_ndvi_raster(stacked)
        else:
            image_stack = stacked
        # 'constant' is the label for label raster.
        labels = one_hot(inputs.get(MODE), n_classes=4)
        labels = tf.cast(labels, tf.int32)
        return image_stack, labels

    return to_tup


def add_ndvi_raster(image_stack):
    '''
    These indices are hardcoded, and taken from the
    sorted keys in feature_spec.
    (NIR - Red) / (NIR + Red)
        2 0_nir_mean
        3 0_red_mean
        8 1_nir_mean
        9 1_red_mean
        14 2_nir_mean
        15 2_red_mean
        20 3_nir_mean
        21 3_red_mean
        26 4_nir_mean
        27 4_red_mean
        32 5_nir_mean
        33 5_red_mean
    '''
    out = []
    for nir_idx, red_idx in NDVI_INDICES:
        # Add a small constant in the denominator to ensure
        # NaNs don't occur because of missing data. Missing
        # data (i.e. Landsat 7 scan line failure) is represented as 0
        # in TFRecord files. Adding \{epsilon} will barely
        # change the non-missing data, and will make sure missing data
        # is still 0 when it's fed into the model.
        ndvi = (image_stack[:, :, nir_idx] - image_stack[:, :, red_idx]) / \
               (image_stack[:, :, nir_idx] + image_stack[:, :, red_idx] + 1e-8)
        out.append(ndvi)
    stack = tf.concat((image_stack, tf.stack(out, axis=-1)), axis=-1)
    return stack


def one_hot(labels, n_classes):
    h, w = labels.shape
    labels = tf.squeeze(labels) - 1
    ls = []
    for i in range(n_classes):
        where = tf.where(labels != i + 1, tf.zeros((h, w)), 1 * tf.ones((h, w)))
        ls.append(where)
    temp = tf.stack(ls, axis=-1)
    return temp


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
