import torch
import matplotlib.pyplot as plt
import imgaug.augmenters as aug
from src.DataAugmentation.Data_Augmentation import AugmentationScheme as AS
import numpy as np
from PIL import Image

def augment_batch(images, labels):

    reformatted_images_list = []
    for i in images:
        reformatted_images_list.append(i[0])

    reformatted_images = np.asarray(reformatted_images_list)

    augSc = AS(reformatted_images, labels)
    augSc.addAugmentation("Flip_lr", percent=1)
    augSc.addAugmentation("Rotate", lo=-45, hi=45)
    augmented_batch, aug_labels = augSc.augmentImages()


    displayImage(reformatted_images[0])
    displayImage(augmented_batch[0])

    augmented_batch = augmented_batch.reshape(64, 1, 28, 28)

    t_augmented_images = torch.from_numpy(augmented_batch)
    t_labels = torch.from_numpy(labels)

    return t_augmented_images, t_labels


def displayImage(image):

    # Image must be numpy array
    plt.imshow(image, cmap='gray')
    plt.show()
