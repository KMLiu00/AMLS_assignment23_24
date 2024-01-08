import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import img_to_array, array_to_img

# set path
data_path = '../Datasets/'

# load data
data = np.load(os.path.join(data_path, 'pathmnist.npz'))
# print data names
for i in data.keys():
    print(i)

# setup data
train_images = data['train_images']
val_images = data['val_images']
test_images = data['test_images']
train_labels = data['train_labels']
val_labels = data['val_labels']
test_labels = data['test_labels']

# transfer from 28*28 to 32*32
def preprocess_images(img, target_size=(32, 32)):
    img = img.astype('float32')
    img = np.array([img_to_array(array_to_img(index, scale=False).resize(target_size)) for index in img])
    img /= 255

    return img


train_images = preprocess_images(train_images)
val_images = preprocess_images(val_images)
test_images = preprocess_images(test_images)

# one-hot encoding
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

print(train_images.shape, val_images.shape, test_images.shape)  # check image shape

# Resnet50 base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# freeze ResNet50 for testing
#for layer in base_model.layers:
   # layer.trainable = False

# tuning layer
x = base_model.output  # connect Resnet50 with tuning layer
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='leaky_relu')(x)
predictions = Dense(9, activation='softmax')(x)  # 9 classes

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# define optimizer, loss, metrics
model.compile(
    optimizer = Adam (learning_rate=0.0005),  # optimizer
    loss='categorical_crossentropy',   # loss function
    metrics=['accuracy']  # metrics function
)

# show parameters after each layer
model.summary()

# train the model
history = model.fit(
    train_images, train_labels,
    batch_size=64,
    validation_data=(val_images, val_labels),
    epochs=20,
)

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
