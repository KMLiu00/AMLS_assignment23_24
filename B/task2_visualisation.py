# import Math Tools
import matplotlib.pyplot as plt
import numpy as np

# import PneumoniaMNIST
from medmnist import PathMNIST


# load train dataset
train_dataset = PathMNIST(split='train')

# print the structure of train_dataset
print(train_dataset)

# load train dataset images and labels
images, labels = train_dataset.imgs, train_dataset.labels

# convert matrix to a single array
labels = labels.flatten()

# take out the first ten numbers from each label
adipose_images = images[labels == 0][:10]
background_images = images[labels == 1][:10]
debris_images = images[labels == 2][:10]
lymphocytes_images = images[labels == 3][:10]
mucus_images = images[labels == 4][:10]
smoothmuscle_images = images[labels == 5][:10]
normalcolonmucosa_images = images[labels == 6][:10]
cancerassociatedstroma_images = images[labels == 7][:10]
colorectaladenocarcinomaepithelium_images = images[labels == 8][:10]

# print data images
plt.figure(figsize=(55, 20))  # Set the figure size to 20 inches wide and 4 inches tall

for i in range(10):  # 10 times loop for printing 10 images per row
    plt.subplot(9, 10, i + 1)  # 2 rows, 10 columns of subplots, and the current subplot being defined is the first row of subplot
    plt.imshow(adipose_images[i].squeeze())  # remove single-dimensional entries from the shape of the array
    # from dataset description, they are gray scale images
    plt.title("Adipose")
    plt.axis('off')  # turn off label

    plt.subplot(9, 10, i + 11)
    plt.imshow(background_images[i].squeeze())
    plt.title("Background")
    plt.axis('off')

    plt.subplot(9, 10, i + 21)
    plt.imshow(debris_images[i].squeeze())
    plt.title("Debris")
    plt.axis('off')

    plt.subplot(9, 10, i + 31)
    plt.imshow(lymphocytes_images[i].squeeze())
    plt.title("Lymphocytes")
    plt.axis('off')

    plt.subplot(9, 10, i + 41)
    plt.imshow(mucus_images[i].squeeze())
    plt.title("Mucus")
    plt.axis('off')

    plt.subplot(9, 10, i + 51)
    plt.imshow(smoothmuscle_images[i].squeeze())
    plt.title("Smooth muscle")
    plt.axis('off')

    plt.subplot(9, 10, i + 61)
    plt.imshow(normalcolonmucosa_images[i].squeeze())
    plt.title("Normal colon mucosa")
    plt.axis('off')

    plt.subplot(9, 10, i + 71)
    plt.imshow(cancerassociatedstroma_images[i].squeeze())
    plt.title("Cancer-associated stroma")
    plt.axis('off')

    plt.subplot(9, 10, i + 81)
    plt.imshow(background_images[i].squeeze())
    plt.title("colorectal adenocarcinoma epithelium")
    plt.axis('off')

plt.show()

