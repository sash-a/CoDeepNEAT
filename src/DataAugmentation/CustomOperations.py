import cv2
import numpy as np


# Any and all custom augmentation operations are defined in this class
class CustomOperation:

    # Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal
    # are sure to be non-edges, so discarded. Those who lie between these two thresholds are classified edges or
    # non-edges based on their connectivity. If they are connected to “sure-edge” pixels,
    # they are considered to be part of edges. Otherwise, they are also discarded.
    class Edges:
        def __init__(self, min_value, max_value):
            self.min_value = min_value
            self.max_value = max_value

        def __call__(self, images, random_state, parents, hooks):
            new_imgs = []

            for img in images:
                # extract edges from image
                edges = cv2.Canny(img, self.min_value, self.max_value)
                rgb_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                # add images to list
                new_imgs.append(rgb_edges)

            # Convert list to numpy array
            aug_imgs = np.asarray(new_imgs)
            return aug_imgs
