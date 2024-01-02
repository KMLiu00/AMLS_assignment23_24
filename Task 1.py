
from medmnist import PneumoniaMNIST
import matplotlib.pyplot as plt
import numpy as np


pneumonia_data = PneumoniaMNIST(split='train', download=True)


images, labels = pneumonia_data.imgs, pneumonia_data.labels

labels = labels.flatten()


normal_images = images[labels == 0][:10]
pneumonia_images = images[labels == 1][:10]


plt.figure(figsize=(20, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(normal_images[i].squeeze(), cmap='gray')
    plt.title("Normal")
    plt.axis('off')

    plt.subplot(2, 10, i+11)
    plt.imshow(pneumonia_images[i].squeeze(), cmap='gray')
    plt.title("Pneumonia")
    plt.axis('off')
plt.show()


print(len(pneumonia_data))
sample_image, sample_label = pneumonia_data[0]
sample_image = np.array(sample_image)
print("Sample label:", sample_label)
print("Shape of sample image:", sample_image.shape)

train_dataset = PneumoniaMNIST(split='train')
val_dataset = PneumoniaMNIST(split='val')
test_dataset = PneumoniaMNIST(split='test')

train_size = len(train_dataset)
val_size = len(val_dataset)
test_size = len(test_dataset)

print("Training set size:", train_size)
print("Validation set size:", val_size)
print("Test set size:", test_size)

