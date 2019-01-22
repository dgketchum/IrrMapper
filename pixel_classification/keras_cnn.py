import h5py
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
    x_train, x_test, y_train, y_test = train_test_split(features, labels,
            test_size=0.1, random_state=42)
    model.fit(x_train,
             y_train,
             batch_size=batch_size,
             epochs=epochs,
             validation_data=(x_test, y_test))
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


if __name__ == '__main__':
    train_dir = 'training_data/'
    
    n_epochs = 40 
    model = keras_model(41, 4)

    for i in range(n_epochs):

        random_indices = np.random.choice(total_instances, total_instances, repeat=False)



        if i > 0:
            evaluate_model(features, labels)

        train_next_batch(model, features, labels, epochs=1)




