import tensorflow as tf 
import numpy as np 
import tensorflow.keras as keras 

#
np.random.seed(42)

# network and training 
EPOCHS = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
RESHAPED = 784

# loading mnist dataset
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], RESHAPED).astype('float32')
X_test = X_test.reshape(X_test.shape[0], RESHAPED).astype('float32')

# normalize in [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

print('train samples : {} | test samples : {} \n'.format(X_train.shape[0], X_test.shape[0]))

# one-hot code target
y_train = keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = keras.utils.to_categorical(y_test, NB_CLASSES)

# make model
model = keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN, input_shape=(RESHAPED,), name='dense_layer', activation='relu'))
model.add(keras.layers.Dense(N_HIDDEN, name='dense_layer1', activation='relu'))
model.add(keras.layers.Dense(NB_CLASSES, name='dense_layer2', activation='softmax'))

# print out model
model.summary()

# train model
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE,
        validation_split=VALIDATION_SPLIT)