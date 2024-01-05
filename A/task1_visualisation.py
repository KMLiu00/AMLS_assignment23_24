# import Math Tools
import matplotlib.pyplot as plt
import numpy as np

# import PneumoniaMNIST
from medmnist import PneumoniaMNIST


# load train dataset
train_dataset = PneumoniaMNIST(split='train', download=True)

# print the structure of train_dataset
print(train_dataset)


# load train dataset images and labels
images, labels = train_dataset.imgs, train_dataset.labels

# convert matrix to a single array
labels = labels.flatten()

# take out the first ten numbers from each label
normal_images = images[labels == 0][:10]
pneumonia_images = images[labels == 1][:10]

# print data images
plt.figure(figsize=(20, 4))  # Set the figure size to 20 inches wide and 4 inches tall
for i in range(10):  # 10 times loop for printing 10 images per row
    plt.subplot(2, 10, i+1)  # 2 rows, 10 columns of subplots, and the current subplot being defined is the first row of subplot
    plt.imshow(normal_images[i].squeeze(), cmap='gray')  # 'squeeze' remove single-dimensional entries from the shape of the array
    plt.title("Normal")
    plt.axis('off')  # turn off label

    plt.subplot(2, 10, i+11)
    plt.imshow(pneumonia_images[i].squeeze(), cmap='gray')
    plt.title("Pneumonia")
    plt.axis('off')
plt.show()


