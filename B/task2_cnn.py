# import PneumoniaMNIST
from medmnist import PneumoniaMNIST

# import Math Tools
import numpy as np

# import TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# import early stopping
from tensorflow.keras.callbacks import EarlyStopping

# import regularization
from tensorflow.keras.regularizers import l2


# function for process data
def preprocess_dataset(dataset):
    dataset = PathMNIST(split=split, download=False, root=path)
    images = []
    labels = []

    for image, label in dataset:  # loop to process all elements in the dataset, no need for label so use _ instead
        image = np.array(image)  # convert image to numpy array format for maximum inter-operability with python
        image = image.astype("float32") / 255  # normalize pixel data from 0-255 to 0-1
        images.append(image)
        labels.append(label)

    images = np.array(images)  # convert the list of image to numpy format
    labels = np.array(labels)  # extract the list of labels to numpy format

    return images, labels


# convolutional neural network
def build_model():
    model = models.Sequential([

        # first layer, 32 filters, each with 3*3 kernel, ReLU activation function,
        # L2 regularization with a coefficient of 0.001,  2*2 pool size.
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 3), kernel_regularizer=l2(0)),
        layers.MaxPooling2D((2, 2)),

        # second layer
        layers.Conv2D(32, (3, 3), activation='relu'),

        # third layer
        layers.Conv2D(32, (3, 3), activation='relu'),

        # forth layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # flatten layer, flattens the input to a one-dimensional array
        layers.Flatten(),

        #  sets randomly set 50% neurons to zero
        layers.Dropout(0),

        # dense Layer with regularization
        layers.Dense(64, activation='relu', kernel_regularizer=l2(0)),

        # output layer, 1 output for binary classification
        layers.Dense(9, activation='softmax')

    ])

    # define optimizer, loss, metrics
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # optimizer
        loss='sparse_categorical_crossentropy',  # loss function
        metrics=['accuracy']  # metrics function
    )


# early stopper, if validation accuracy stop improving for 8 times，it will stop the training
early_stopper = EarlyStopping(monitor='val_loss', patience=0)


# train the model

def run(path):
    train_images, train_labels = preprocess_dataset(path, 'train')
    val_images, val_labels = preprocess_dataset(path, 'val')
    test_images, test_labels = preprocess_dataset(path, 'test')

    model = build_model()

    early_stopper = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        train_images, train_labels, epochs=50,
        validation_data=(val_images, val_labels),
        batch_size=128,  # batch_size
        callbacks=[early_stopper]  # callback for early stopping
    )

    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    return history.history, test_loss, test_accuracy

