# import Math Tools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
# import TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models

# import early stopping
from tensorflow.keras.callbacks import EarlyStopping

# import regularization
from tensorflow.keras.regularizers import l1, l2

# import PathMNIST
from medmnist import PathMNIST
data_flag = 'pathmnist'
download = True

# load train dataset
train_dataset = PathMNIST(split='train', download=download)
# validation train dataset
val_dataset = PathMNIST(split='val', download=download)
# test dataset
test_dataset = PathMNIST(split='test', download=download)


# function for process data
def preprocess_dataset(dataset):
    images = []
    labels = []

    for image, _ in dataset:  # loop to process all elements in the dataset, no need for label so use _ instead
        image = np.array(image)  # convert image to numpy array format for maximum inter-operability with python
        image = image.astype("float32") / 255  # normalize pixel data from 0-255 to 0-1
        images.append(image)  # appends the normalized image to the images list.

    images = np.array(images)  # convert the list of image to numpy array

    labels = np.array([label for _, label in dataset])  # extract labels from the dataset and convert to a numpy array

    return images, labels


# process all data
train_images, train_labels = preprocess_dataset(train_dataset)
val_images, val_labels = preprocess_dataset(val_dataset)
test_images, test_labels = preprocess_dataset(test_dataset)

# convolutional neural network
model = models.Sequential([

    # first layer, 32 filters, each with 3*3 kernel, ReLU activation function,
    # L2 regularization with a coefficient of 0.001,  2*2 pool size.
    layers.Conv2D(64, (3, 3), activation='leaky_relu', input_shape=(28, 28, 3), kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D((2, 2)),

    # second layer
    layers.Conv2D(128, (3, 3), activation='leaky_relu'),
    layers.MaxPooling2D((2, 2)),

    # third layer
    layers.Conv2D(128, (3, 3), activation='leaky_relu'),
    layers.MaxPooling2D((2, 2)),

    # flatten layer, flattens the input to a one-dimensional array
    layers.Flatten(),

    #  sets randomly set 70% neurons to zero
    layers.Dropout(0.70),

    # dense Layer with regularization
    layers.Dense(128, activation='leaky_relu', kernel_regularizer=l2(0.001)),

    # output layer, 1 output for binary classification
    layers.Dense(9, activation='softmax')
])

# show parameters after each layer
model.summary()

# define optimizer, loss, metrics
model.compile(
    optimizer = Adam (learning_rate=0.0005), # optimizer
    loss='sparse_categorical_crossentropy',  # loss function
    metrics=['accuracy']  # metrics function
)

# early stopper, if validation accuracy stop improving for 8 times，it will stop the training
early_stopper = EarlyStopping(monitor='val_loss', patience=8)

# train the model
history = model.fit(
    train_images, train_labels,
    epochs=80,  # go over the full dataset for 80 times, but it is controlled by early stop
    validation_data=(val_images, val_labels),
    batch_size=128,  # batch_size
    callbacks=[early_stopper]  # callback for early stopping
)

# use independent test set for testing
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_accuracy)
print("Test loss:", test_loss)

# plotting
# extract train accuracy and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# extract train loss and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)  # length is from 0, so need to add 1

# plot accuracy curve
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# plot loss curve
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
