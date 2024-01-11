# import PneumoniaMNIST
from keras.utils import to_categorical
from medmnist import PathMNIST

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
    dataset = PathMNIST(split=split, download=False, root=path)
    images = []
    labels = []

    for image, label in dataset:
        image = np.array(image)
        image = image.astype("float32") / 255
        images.append(image)
        labels.append(label)

    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=9)

    return images, labels


def build_model():
    model = models.Sequential([

        # first layer, 32 filters, each with 3*3 kernel, ReLU activation function,
        # L2 regularization with a coefficient of 0.001,  2*2 pool size.
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 3), kernel_regularizer=l2(0)),
        layers.MaxPooling2D((2, 2)),

        # second layer
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # third layer
        layers.Conv2D(128, (3, 3), activation='relu'),
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

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def run(path):
    train_images, train_labels = preprocess_dataset(path, 'train')
    val_images, val_labels = preprocess_dataset(path, 'val')
    test_images, test_labels = preprocess_dataset(path, 'test')

    model = build_model()

    early_stopper = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        train_images, train_labels, epochs=50,
        validation_data=(val_images, val_labels),
        batch_size=128,
        callbacks=[early_stopper]
                        )

    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    # 返回训练历史和测试结果
    return history.history, test_loss, test_accuracy
