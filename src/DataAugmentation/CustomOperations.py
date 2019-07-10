import numpy as np
# Any and all custom augmentation operations are defined in this class
class CustomOperation:

    # Replace in every image each fourth row with black pixels:
    def customFunc1(self, images, random_state, parents, hooks):
        for img in images:
            img[::4] = 0
        return images

    # Unsure on why or how this is used but documentation states it is needed for the lambda augmenter
    def keypoint_func(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images