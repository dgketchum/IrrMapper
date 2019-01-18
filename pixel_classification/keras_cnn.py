import h5py
from glob import glob
import tensorflow as tf
import numpy as np
from shuffle_data import next_batch

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
    return model

def train_model(kernel_size, features, labels, n_classes=4):

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(features, labels,
            test_size=0.1, random_state=42)

    model = keras_model(kernel_size, n_classes)
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    model.fit(x_train,
             y_train,
             batch_size=128,
             epochs=10,
             validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])
    return model

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


if __name__ == '__main__':
    train_dir = 'training_data/'
    irrigated = ['MT_Sun_River_2013', "MT_Huntley_Main_2013"]
    other = ['other']
    fallow = ['Fallow']
    forest = ['Forrest']
    n = 10000
    irr = {'files':[f for f in glob(train_dir + "*.h5") if is_it(f, irrigated)], 'instances':n}
    fall = {'files':[f for f in glob(train_dir + "*.h5") if is_it(f, fallow)], 'instances':n}
    forest_ = {'files':[f for f in glob(train_dir + "*.h5") if is_it(f, forest)], 'instances':n}
    other_ = {'files':[f for f in glob(train_dir + "*.h5") if is_it(f, other)], 'instances':n}
    
    #fall = [f for f in glob(shp_dir) if is_it(f, fallow)]
    #forest_ = [f for f in glob(shp_dir) if is_it(f, forest)]
    #other_ = [f for f in glob(shp_dir) if is_it(f, other)]

    file_map = {0: irr, 1:fall, 2:forest_, 3:other_}

    for i in range(2):
        features, labels = get_next_batch(file_map)
        print(features.shape, labels.shape)
        train_model(41, features, labels)

