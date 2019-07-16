import torch
import matplotlib.pyplot as plt
from src.DataAugmentation.AugmentationScheme import AugmentationScheme as AS
import numpy as np
import random


# augments batches of images
def augment_batch(images, labels, augmentor: AS):
    batch_size = np.shape(images)[0]
    channels = np.shape(images)[1]
    x_Dim = np.shape(images)[2]
    y_dim = np.shape(images)[3]

    # Create a new array that contains the batch of images that have been reformatted to accommodate the aug lib
    reformatted_images_list = []
    for i in images:
        reformatted_images_list.append(i[0])
    reformatted_images = np.asarray(reformatted_images_list)

    # Create instance of DA class in order to create desired augmentation scene
    # augSc = AS(reformatted_images, labels)
    # Choose desired augmentations
    # augSc.add_augmentation("Flip_lr", percent=1)
    # augSc.add_augmentation("Rotate", lo=-45, hi=45)
    # Create augmented batch of images and labels

    augmentor.images = reformatted_images
    augmentor.labels = labels
    augmented_batch, aug_labels = augmentor.augment_images()

    # Displays original image + augmented image (for testing)
    if random.random() < 0.001:
        display_image(reformatted_images[0])
        display_image(augmented_batch[0])

    # Reformat augmented batch into the shape that the  rest of the code wants
    augmented_batch = augmented_batch.reshape(batch_size, channels, x_Dim, y_dim)

    # Convert images stored in numpy arrays to tensors
    t_augmented_images = torch.from_numpy(augmented_batch)
    t_labels = torch.from_numpy(labels)

    return t_augmented_images, t_labels


def display_image(image):
    # Image must be numpy array
    plt.imshow(image, cmap='gray')
    plt.show()
