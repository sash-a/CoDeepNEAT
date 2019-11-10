import imgaug.augmenters as iaa
import numpy as np
import random


# The AugmenationScheme class allows for the creation of a augmentation schemes.

class AugmentationScheme:
    # 'augs' is a list that contains all data augmentations in the scheme
    def __init__(self, augs):
        print("augmentation scheme created")
        self.augs = augs

    def __call__(self, image):
        image = np.array(image)
        aug_scheme = iaa.Sometimes(0.5, iaa.SomeOf(random.randrange(1, len(self.augs)+1), self.augs, random_order=True))
        aug_img = aug_scheme.augment_image(image)
        # fixes negative strides
        aug_img = aug_img[..., ::1] - np.zeros_like(aug_img)
        return aug_img
