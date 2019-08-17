import torch
import matplotlib.pyplot as plt
from src.DataAugmentation.AugmentationScheme import AugmentationScheme as AS
import numpy as np
import random
import cv2

from src.Config import Config

# augments batches of images
def augment_batch(images, labels, augmentor: AS):

    batch_size = np.shape(images)[0]
    channels = np.shape(images)[1]
    x_Dim = np.shape(images)[2]
    y_dim = np.shape(images)[3]

    reformatted_images = reformat_images_for_DA(images)

    augmentor.images = reformatted_images
    augmentor.labels = labels
    augmented_batch, aug_labels = augmentor.augment_images()

    # Displays original image + augmented image (for testing)
    # if random.random() < 1:
    #     print("DA's:",augmentor.augs)
    #     display_image(reformatted_images[0])
    #     display_image(augmented_batch[0])

    # convert augmented images back to dtype float32
    reformatted_augmented_batch = reformat_images_for_system(augmented_batch)

    # Reformat augmented batch into the shape that the rest of the code wants
    reformatted_augmented_batch = reformatted_augmented_batch.reshape(batch_size, channels, x_Dim, y_dim)

    # Convert images stored in numpy arrays to tensors
    t_augmented_images = torch.from_numpy(reformatted_augmented_batch)
    t_labels = torch.from_numpy(labels)

    norm_t_augmented_images = (t_augmented_images/255)*2-1

    return norm_t_augmented_images, t_labels


def display_image(image):
    if Config.colour_augmentations:
        # image = image / 2 + 0.5   # un-normalize
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')

    plt.show()


# Creates a new array that contains the batch of images that have been reformatted to accommodate the aug lib
def reformat_images_for_DA(images):
    # different reformatting required based on if images are 1 channel or RGB
    if Config.colour_augmentations:
        reformatted_images_list = []
        for i in images:
            # convert image to uint8 (NB for certain DAs)
            img = norm8(i)
            # reshuffles dimensions of image to accommodate the library
            img = np.transpose(img, (1, 2, 0))
            # adds image to list
            reformatted_images_list.append(img)
        # converts to numpy
        reformatted_images = np.asarray(reformatted_images_list)
        return reformatted_images
    else:
        reformatted_images_list = []
        for i in images:
            # convert 1 channel image to 3 channel image
            rgb = cv2.cvtColor(i[0], cv2.COLOR_GRAY2RGB)
            # convert image to uint8 (NB for certain DAs)
            img = norm8(rgb)
            reformatted_images_list.append(img)
        reformatted_images = np.asarray(reformatted_images_list)
        return reformatted_images


def reformat_images_for_system(augmented_batch):
    reformatted_augmented_batch_list = []
    if Config.colour_augmentations:
        for img in augmented_batch:
            reformatted_augmented_batch_list.append(float32(img))
        reformatted_augmented_batch = np.asarray(reformatted_augmented_batch_list)
    else:
        for img in augmented_batch:
            # convert 3 channel image to 1 channel image
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            reformatted_augmented_batch_list.append(float32(gray_img))
        reformatted_augmented_batch = np.asarray(reformatted_augmented_batch_list)

    return reformatted_augmented_batch


# convert image to data type uint8 (grayscale and HSV require images to be uint8)
def norm8(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img

# convert image to data type float32 (system requires image to be float32)
def float32(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    return img
