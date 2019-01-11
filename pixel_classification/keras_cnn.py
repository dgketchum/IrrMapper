import h5py
import glob
import tensorflow as tf
import numpy as np

N_INSTANCES_IRRIGATED = 30000
N_INSTANCES_NOT = 10000

def keras_model(kernel_size):
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
        input_shape=(kernel_size, kernel_size, 3))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2)) model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    # Take a look at the model summary
    model.summary()
    return model

def train_model(kernel_size):

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(features, labels,
            test_size=0.1, random_state=42)

    model = keras_model(kernel_size)
    model.compile(loss='binary_crossentropy',
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
        ret[i, e] = 1
    return ret

def generate_labels_and_features(filename, class_code, index_1, index_2, n_classes=2):
    # approach:
    # I have n files containing training data on disk.
    # Loop through all classes and sample a subset
    # of each file. This actually shouldn't be that hard.
    # Then, shuffle the data (in memory?) and split it
    # into training and test sets. 
    with h5py.File(filename, 'r') as f:
        data = f['cc:'+str(class_code)]
    labels = [class_code]*(index_2-index_1)
    labels = make_one_hot(labels, n_classes=n_classes)

    return data[index_1:index_2, :, :, :] # this is an assumption about the shape of the data

def shuffle_data(training_directory, suffix='.h5'):
    # Make piles, and shuffle that way.  
    # Reference that website.  
    # approach:
    # for each (h5) file in directory:
    #  open it, and make piles with it (in parallel)
    #  then combine each litle pile into a large pile, 
    #  but iterate through the littler piles when
    #  creating the big pile 
    return None
    









    















