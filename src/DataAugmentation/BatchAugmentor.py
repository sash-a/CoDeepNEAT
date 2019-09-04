import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.Config import Config
from src.DataAugmentation.AugmentationScheme import AugmentationScheme as AS


# augments batches of images
def augment_batch(images, labels, augmentor: AS):

    # images are reformatted to accommodate the imagaug library
    reformatted_images = reformat_images_for_DA(images, augmentor)

    augmentor.images = reformatted_images
    augmentor.labels = labels
    augmented_batch, aug_labels = augmentor.augment_images()

    # convert augmented images back to dtype float32 (if necessary)
    converted_augmented_batch = convert_images_for_system(augmented_batch, -1, 1, augmentor)
    # reformat images (by transposing) to fit what the pytorch system expects
    reformatted_augmented_batch = reformat_images_for_system(converted_augmented_batch)

    # Convert images stored in numpy arrays to tensors
    t_augmented_images = torch.from_numpy(reformatted_augmented_batch)
    # convert labels to tensors
    t_labels = torch.from_numpy(labels)

    return t_augmented_images, t_labels


def display_image(image):
    """Displays an image"""
    if Config.colour_augmentations:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')

    plt.show()


#
def reformat_images_for_DA(images, augmentor):
    """Returns an array of images that have been reformatted to accommodate the imgaug library"""
    reformatted_images_list = []
    # different reformatting required based on if images are 1 channel or 3 channel (grayscale or RGB)
    if Config.colour_augmentations:

        for i in images:
            # the following DAs require conversion to uint8
            if "Grayscale" in augmentor.augs_names or "HSV" in augmentor.augs_names or "Custom_Canny_Edges" in augmentor.augs_names:
                # convert image to uint8
                img = norm8(i)
            else:
                img = i

            # reshuffles dimensions of image to accommodate the imgaug library
            img = np.transpose(img, (1, 2, 0))
            reformatted_images_list.append(img)

        # converts list to numpy
        reformatted_images = np.asarray(reformatted_images_list)
        return reformatted_images
    else:

        for i in images:
            # convert 1 channel image to 3 channel image (necessary for specific DAs)
            # Also makes transposing these images unnecessary
            rgb = cv2.cvtColor(i[0], cv2.COLOR_GRAY2RGB)
            # The following DAs require conversion to uint8
            if "Grayscale" in augmentor.augs_names or "HSV" in augmentor.augs_names or "Custom_Canny_Edges" in augmentor.augs_names:
                # convert image to uint8
                img = norm8(rgb)
            else:
                img = rgb

            reformatted_images_list.append(img)

        # converts list to numpy
        reformatted_images = np.asarray(reformatted_images_list)
        return reformatted_images


def convert_images_for_system(augmented_batch, start_range, end_range, augmentor):
    """Converts augmented images that are not the correct dtype into float32"""
    reformatted_augmented_batch_list = []
    if Config.colour_augmentations:
        for img in augmented_batch:
            # only data augmentation schemes consisting of these augmentations will need to be converted
            if "Grayscale" in augmentor.augs_names or "HSV" in augmentor.augs_names or "Custom_Canny_Edges" in augmentor.augs_names:
                reformatted_augmented_batch_list.append(float32(img, start_range, end_range))
            else:
                reformatted_augmented_batch_list.append(img)

        reformatted_augmented_batch = np.asarray(reformatted_augmented_batch_list)
    else:
        for img in augmented_batch:
            # convert 3 channel image back to 1 channel image
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # only data augmentation schemes consisting of these augmentations will need to be converted
            if "Grayscale" in augmentor.augs_names or "HSV" in augmentor.augs_names or "Custom_Canny_Edges" in augmentor.augs_names:
                reformatted_augmented_batch_list.append(float32(gray_img, start_range, end_range))
            else:
                reformatted_augmented_batch_list.append(gray_img)

        reformatted_augmented_batch = np.asarray(reformatted_augmented_batch_list)

    return reformatted_augmented_batch


def reformat_images_for_system(augmented_batch):
    """Returns an array of images that have been reformatted to accommodate the pytorch system"""
    reformatted_aug_images = []
    for img in augmented_batch:
        reformatted_aug_images.append(np.transpose(img, (2, 0, 1)))

    return np.asarray(reformatted_aug_images)


# currently, only grayscale, HSV and Canny_Edges require images to be uint8
def norm8(img):
    """Convert image to data type uint8"""
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img


# pytorch system requires images to be float32
def float32(img, start_range, end_range):
    """Convert image to data type float32"""
    img = cv2.normalize(img, None, start_range, end_range, cv2.NORM_MINMAX, cv2.CV_32F)
    return img
