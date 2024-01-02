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
    plt.imshow(normal_images[i].squeeze(), cmap='gray')  # remove single-dimensional entries from the shape of the array
    # from dataset description, they are gray scale images
    plt.title("Normal")
    plt.axis('off')  # turn off label

    plt.subplot(2, 10, i+11)
    plt.imshow(pneumonia_images[i].squeeze(), cmap='gray')
    plt.title("Pneumonia")
    plt.axis('off')
plt.show()

# print train_dataset basic information
print(len(train_dataset))
sample_image, sample_label = train_dataset[0]
sample_image = np.array(sample_image)
print("Sample label:", sample_label)
print("Shape of sample image:", sample_image.shape)

train_dataset = PneumoniaMNIST(split='train')
val_dataset = PneumoniaMNIST(split='val')
test_dataset = PneumoniaMNIST(split='test')

# dataset length with different split
train_size = len(train_dataset)
val_size = len(val_dataset)
test_size = len(test_dataset)

print("Training set size:", train_size)
print("Validation set size:", val_size)
print("Test set size:", test_size)
