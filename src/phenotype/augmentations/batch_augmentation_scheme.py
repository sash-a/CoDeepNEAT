import random
from typing import List

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch

from src.configuration import config


class BatchAugmentationScheme:
    """Allows for imgaug to augment each batch of images"""

    def __init__(self, augs: list):
        self.augs = augs  # augs is a list that contains all data augmentations in the scheme

    def __call__(self, images: list):
        images = self.before(images)
        aug_scheme = iaa.Sometimes(config.apply_da_chance,
                                   iaa.SomeOf(random.randrange(1, len(self.augs) + 1), self.augs, random_order=True))
        images = aug_scheme.augment_images(images)
        images = self.after(images, -1, 1)

        # fixes negative strides
        # aug_img = aug_img[..., ::1] - np.zeros_like(aug_img)

        return torch.from_numpy(np.asarray(images))

    def before(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Returns a nparray of images that have been reformatted to accommodate the imgaug library"""
        for i in range(len(images)):
            if config.use_colour_augmentations:
                images[i] = norm8(images[i])
                images[i] = np.transpose(images[i], (1, 2, 0))
            else:
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_GRAY2RGB)

        return images

    def after(self, images: List[np.ndarray], start_range, end_range) -> List[np.ndarray]:
        """Returns a nparray of images that have been reformatted to accommodate the pytorch library"""
        for i in range(len(images)):
            if config.use_colour_augmentations:
                images[i] = np.transpose(float32(images[i], start_range, end_range), (2, 0, 1))
            else:
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)

        return images


# Augmentor requires images to be uint8
def norm8(img):
    """Convert image to data type uint8"""
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# pytorch system requires images to be float32
def float32(img, start_range, end_range):
    """Convert image to data type float32"""
    return cv2.normalize(img, None, start_range, end_range, cv2.NORM_MINMAX, cv2.CV_32F)
