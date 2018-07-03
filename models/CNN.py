    # -*- coding: utf-8 -*-
"""
Created on Sat Nov 4 16:30:52 2017

@author: SainiD
"""
#Defining the attributes
#batch size to train
batch_size = 128
#number of output classes
nb_classes = 10
#numbe of epochs to train
nb_epoch = 5
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
#size of pooling area for max pooling
nb_pool = 2
#convolution kernel size
nb_conv = 3

# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np 

# the data, shuffled and split between train and test sets
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Reshape the data
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#normalize
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#Consider an instance and check the image mapping
i = 4600
print('Label:', Y_train[i:])

#Training the CNN model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras import backend as K
K.set_image_dim_ordering('th')

model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1,img_rows,img_cols)))
convout1 = Activation('tanh')
model.add(Activation('tanh'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('tanh')
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adadelta')

#Augmentation used for regularization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(zoom_range = 0.1,height_shift_range = 0.1,
                             width_shift_range = 0.1,rotation_range = 10)	

#Model compilation: optimizing using adams
model.compile(loss='categorical_crossentropy', 
              optimizer = Adam(lr=1e-4), metrics=["accuracy"])

#Reduce the learning rate by 10% after every epoch
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

#Trainnig the model with a small validation set(steps per epoch= 50)
hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                           steps_per_epoch=50,
                           epochs=nb_epoch, #Increase this when not on Kaggle kernel
                           verbose=1,  #1 for ETA, 0 for silent
                           validation_data= (X_test, Y_test), #For speed
                           callbacks=[annealer])

#Final Evalution of the trained model
final_loss, final_acc = model.evaluate(X_test, Y_test, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

# Plot for loss and accuracy vs epoch
plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.legend(['Training data', 'validation data'], loc='upper left')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.legend(['Training data','validation data'], loc='upper left')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

#Confussion matrix
from sklearn.metrics import confusion_matrix
y_hat = model.predict(X_test)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(Y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)