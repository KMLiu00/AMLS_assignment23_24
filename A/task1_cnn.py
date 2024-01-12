# import PneumoniaMNIST
from keras.utils import to_categorical
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


def preprocess_dataset(path, split):
    dataset = PneumoniaMNIST(split=split, download=False, root=path)  # already downloaded dataset in main file,
    images = []
    labels = []

    for image, label in dataset:
        image = np.array(image)
        image = image.astype("float32") / 255  # normalize pixel data from 0-255 to 0-1
        images.append(image)
        labels.append(label)

    images = np.array(images)  # convert the list of image to numpy format
    labels = np.array(labels)  # extract the list of labels to numpy format

    return images, labels


def build_model():  # call at function run
    model = models.Sequential([

        # first layer, 9 filters, each with 3*3 kernel, Leaky_ReLU activation function,
        # L2 regularization,  2*2 max pooling size.
        layers.Conv2D(9, (3, 3), activation='leaky_relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.05)),
        layers.MaxPooling2D((2, 2)),

        # second layer
        layers.Conv2D(18, (3, 3), activation='leaky_relu'),
        layers.MaxPooling2D((2, 2)),

        # flatten layer, flat the input to a one-dimensional array
        layers.Flatten(),

        #  set average 50% neurons inactive
        layers.Dropout(0.5),

        # dense Layer with L2 regularization
        layers.Dense(5, activation='leaky_relu', kernel_regularizer=l2(0.051)),

        # output layer, 1 output for binary classification
        layers.Dense(1, activation='sigmoid')
    ])

    # define optimizer, loss, metrics
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # optimizer
        loss='binary_crossentropy',  # loss function
        metrics=['accuracy']  # metrics function
    )
    return model


def run(path):
    # split dataset
    train_images, train_labels = preprocess_dataset(path, 'train')
    val_images, val_labels = preprocess_dataset(path, 'val')
    test_images, test_labels = preprocess_dataset(path, 'test')

    model = build_model()   # call build_model function

    early_stopper = EarlyStopping(monitor='val_loss', patience=5)   # early stopping for continues 5 times increase
    # of val_loss

    history = model.fit(        # train the model
        train_images, train_labels,
        epochs=50,
        validation_data=(val_images, val_labels),
        batch_size=32,  # batch_size
        callbacks=[early_stopper]
        )

    test_loss, test_accuracy = model.evaluate(test_images, test_labels) #

    return history.history, test_loss, test_accuracy

