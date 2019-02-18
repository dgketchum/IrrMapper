import h5py
import os
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from shuffle_data import one_epoch

def keras_model(kernel_size, n_classes):
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
        input_shape=(36, kernel_size, kernel_size))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2)) 
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    # Take a look at the model summary
    model.summary()
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model

def train_next_batch(model, features, labels, n_classes=4, epochs=5, batch_size=128):

    # shuffle the labels again

    tb = TensorBoard(log_dir='graphs/cnn/')
    x_train, x_test, y_train, y_test = train_test_split(features, labels,
            test_size=0.01, random_state=42)
    model.fit(x_train,
             y_train,
             batch_size=batch_size,
             epochs=epochs,
             validation_data=(x_test, y_test),
             callbacks=[tb])
    return model


def evaluate_model(features, labels):
    score = model.evaluate(features, labels, verbose=0)
    print('\n', 'Test accuracy:', score[1], '\n')

def make_one_hot(labels, n_classes):
    ret = np.zeros((len(labels), n_classes))
    for i, e in enumerate(labels):
        ret[i, int(e)] = 1
    return ret

def get_next_batch(file_map, n_classes=4):
    features, labels = next_batch(file_map)
    labels = make_one_hot(labels, n_classes)
    return features, labels

def is_it(f, targets):
    for e in targets:
        if e in f and 'sample' not in f:
            return True
    return False

def fnames(class_code):
    return "training_data/class_{}_train.h5".format(class_code)

# Yield the concatenated training array?

if __name__ == '__main__':
    train_dir = 'training_data/'
    model_dir = 'models/'
    n_epochs = 1
    kernel_size = 41
    model_name = 'model_kernel_{}'.format(kernel_size)
    total_instances = 100000
    
    model_path = os.path.join(model_dir, model_name)
    model = keras_model(41, 2)
    model = tf.keras.models.load_model(model_path)
    features = np.zeros((128, 36, 41, 41))
    labels = np.zeros((128, 4))
    train_next_batch(model, features, labels)
    if not os.path.isfile(model_path):
        model.save(model_path)


