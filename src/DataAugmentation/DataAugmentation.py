# The imgaug library is integral to this class (must cite)
import imgaug.augmenters as iaa
from src.DataAugmentation.CustomOperations import CustomOperation
import copy


# The AugmenationScheme class allows for the creation of a pipeline consiting of different augmentations.
# Once all the desired augmentations are chosen for the pipeline.
# The pipeline can be used to create a DA scheme for a given image set


class AugmentationScheme:

    # Upon initialisation we create our pipeline which will eventually become our DA scheme
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.augs = []

        # Dictionary containing all possible augmentations functions
        self.Augmentations = {

            # WithColorspace: Apply child augmenters within a specific color space:

            # Convert images to HSV, then increase each pixel's Hue (H), Saturation (S) or Value/lightness (V) [0, 1, 2]
            # value by an amount in between lo and hi:
            "HSV": lambda channel, lo, hi: iaa.WithColorspace
            (to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(channel, iaa.Add((lo, hi)))),

            # WithChannels: Apply child augmenters to specific channels:

            # Increase each pixel’s channel-value (redness/greenness/blueness) [0, 1, 2] by value in between lo and hi:
            "Increase_Channel": lambda channel, lo, hi: iaa.WithChannels(channel, iaa.Add((lo, hi))),
            # Rotate each image’s channel [R=0, G=1, B=2] by value in between lo and hi degrees:
            "Rotate_Channel": lambda channel, lo, hi: iaa.WithChannels(channel, iaa.Affine(rotate=(lo, hi))),

            # Augmenter that never changes input images (“no operation”).
            "No_Operation": iaa.Noop(),

            # Pads images, i.e. adds columns/rows to them. Pads image by value in between lo and hi
            # percent relative to its original size (only accepts positive values in range[0, 1]):
            # NOTE: automatically resizes images back to their original size after it has augmented them.
            "Pad": lambda lo, hi: iaa.Pad(percent=(lo, hi)),

            # Crops/cuts away pixels at the sides of the image.
            # Crops images by value in between lo and hi (only accepts positive values in range[0, 1]):
            # NOTE: automatically resizes images back to their original size after it has augmented them.
            "Crop": lambda lo, hi: iaa.Crop(percent=(lo, hi)),

            # Flip/mirror percent (i.e 0.5) of the input images horizontally
            # The default probability is 0, so to flip all images, percent=1
            "Flip_lr": lambda percent: iaa.Fliplr(percent),

            # Flip/mirror percent (i.e 0.5) of the input images vertically
            # The default probability is 0, so to flip all images, percent=1
            "Flip_ud": lambda percent: iaa.Flipud(percent),

            # Completely or partially transform images to their superpixel representation.
            # Generate s_pix_lo to s_pix_hi superpixels per image. Replace each superpixel with a probability between
            # prob_lo and prob_hi with range[0, 1] (sampled once per image) by its average pixel color.
            "Superpixels": lambda prob_lo, prob_hi, s_pix_lo, s_pix_hi:
            iaa.Superpixels(p_replace=(prob_lo, prob_hi), n_segments=(s_pix_lo, s_pix_hi)),

            # Change images to grayscale and overlay them with the original image by varying strengths,
            # effectively removing alpha_lo to alpha_hi of the color:
            "Grayscale": lambda alpha_lo, alpha_hi: iaa.Grayscale(alpha=(alpha_lo, alpha_hi)),

            # Blur each image with a gaussian kernel with a sigma between sigma_lo and sigma_hi:
            "Gaussian_Blur": lambda sigma_lo, sigma_hi: iaa.GaussianBlur(sigma=(sigma_lo, sigma_hi)),

            # Blur each image using a mean over neighbourhoods that have random sizes,
            # which can vary between h_lo and h_hi in height and w_lo and w_hi in width:
            "Average_Blur": lambda h_lo, h_hi, w_lo, w_hi: iaa.AverageBlur(k=((h_lo, h_hi), (w_lo, w_hi))),

            # Blur each image using a median over neighbourhoods that have a random size between lo x lo and hi x hi:
            "Median_Blur": lambda lo, hi: iaa.MedianBlur(k=(lo, hi)),

            # Sharpen an image, then overlay the results with the original using an alpha between alpha_lo and alpha_hi:
            "Sharpen": lambda alpha_lo, alpha_hi, lightness_lo, lightness_hi: iaa.Sharpen
            (alpha=(alpha_lo, alpha_hi), lightness=(lightness_lo, lightness_hi)),

            # Emboss an image, then overlay the results with the original using an alpha between alpha_lo and alpha_hi:
            "Emboss": lambda alpha_lo, alpha_hi, strength_lo, strength_hi:
            iaa.Emboss(alpha=(alpha_lo, alpha_hi), strength=(strength_lo, strength_hi)),

            # Detect edges in images, turning them into black and white images and
            # then overlay these with the original images using random alphas between alpha_lo and alpha_hi:
            "Detect_Edges": lambda alpha_lo, alpha_hi: iaa.EdgeDetect(alpha=(alpha_lo, alpha_hi)),

            # Detect edges having random directions between dir_lo and dir_hi (i.e (0.0, 1.0) = 0 to 360 degrees) in
            # images, turning the images into black and white versions and then overlay these with the original images
            # using random alphas between alpha_lo and alpha_hi:
            "Directed_edge_Detect": lambda alpha_lo, alpha_hi, dir_lo, dir_hi:
            iaa.DirectedEdgeDetect(alpha=(alpha_lo, alpha_hi), direction=(dir_lo, dir_hi)),

            # Add random values between lo and hi to images. In percent of all images the values differ per channel
            # (3 sampled value). In the rest of the images the value is the same for all channels:
            "Add": lambda lo, hi, percent: iaa.Add((lo, hi), per_channel=percent),

            # Adds random values between lo and hi to images, with each value being sampled per pixel.
            # In percent of all images the values differ per channel (3 sampled value). In the rest of the images
            # the value is the same for all channels:
            "Add_Element_Wise": lambda lo, hi, percent: iaa.AddElementwise((lo, hi), per_channel=percent),

            # Add gaussian noise (aka white noise) to an image, sampled once per pixel from a normal
            # distribution N(0, s), where s is sampled per image and varies between lo and hi*255 for percent of all
            # images and sampled three times (channel-wise) for the rest from the same normal distribution:
            "Additive_Gaussian_Noise": lambda lo, hi, percent:
            iaa.AdditiveGaussianNoise(scale=(lo, hi * 255), per_channel=percent),

            # Multiply in percent of all images each pixel with random values between lo and hi and multiply
            # the pixels in the rest of the images channel-wise,
            # i.e. sample one multiplier independently per channel and pixel:
            "Multiply": lambda lo, hi, percent: iaa.Multiply((lo, hi), per_channel=percent),

            # Multiply values of pixels with possibly different values for neighbouring pixels,
            # making each pixel darker or brighter. Multiply each pixel with a random value between lo and hi:
            "Multiply_Element_Wise": lambda lo, hi, percent: iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),

            # Augmenter that sets a certain fraction of pixels in images to zero.
            # Sample per image a value p from the range lo<=p<=hi and then drop p percent of all pixels in the image
            # (i.e. convert them to black pixels), but do this independently per channel in percent of all images
            "Dropout": lambda lo, hi, percent: iaa.Dropout(p=(lo, hi), per_channel=percent),

            # Augmenter that sets rectangular areas within images to zero.
            # Drop d_lo to d_hi percent of all pixels by converting them to black pixels,
            # but do that on a lower-resolution version of the image that has s_lo to s_hi percent of the original size,
            # Also do this in percent of all images channel-wise, so that only the information of some
            # channels is set to 0 while others remain untouched:
            "Course_Dropout": lambda d_lo, d_hi, s_lo, s_hi, percent:
            iaa.CoarseDropout((d_lo, d_hi), size_percent=(s_hi, s_hi), per_channel=percent),

            # Augmenter that inverts all values in images, i.e. sets a pixel from value v to 255-v.
            # For c_percent of all images, invert all pixels in these images channel-wise with probability=i_percent
            # (per image). In the rest of the images, invert i_percent of all channels:
            "Invert": lambda i_percent, c_percent: iaa.Invert(i_percent, per_channel=c_percent),

            # Augmenter that changes the contrast of images.
            # Normalize contrast by a factor of lo to hi, sampled randomly per image
            # and for percent of all images also independently per channel:
            "Contrast_Normalisation": lambda lo, hi, percent: iaa.ContrastNormalization((lo, hi), per_channel=percent),

            # Scale images to a value of lo to hi percent of their original size but do this independently per axis:
            "Scale": lambda x_lo, x_hi, y_lo, y_hi: iaa.Affine(scale={"x": (x_lo, x_hi), "y": (y_lo, y_hi)}),

            # Translate images by lo to hi percent on x-axis and y-axis independently:
            "Translate_Percent": lambda lo, hi: iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),

            # Translate images by lo to hi pixels on x-axis and y-axis independently:
            "Translate_Pixels": lambda x_lo, x_hi, y_lo, y_hi:
            iaa.Affine(translate_px={"x": (x_lo, x_hi), "y": (y_lo, y_hi)}),

            # Rotate images by lo to hi degrees:
            "Rotate": lambda lo, hi: iaa.Affine(rotate=(lo, hi)),

            # Shear images by lo to hi degrees:
            "Shear": lambda lo, hi: iaa.Affine(shear=(lo, hi)),

            # Augmenter that places a regular grid of points on an image and randomly moves the neighbourhood of
            # these point around via affine transformations. This leads to local distortions.
            # Distort images locally by moving points around, each with a distance v (percent relative to image size),
            # where v is sampled per point from N(0, z) z is sampled per image from the range lo to hi:
            "Piecewise_Affine": lambda lo, hi: iaa.PiecewiseAffine(scale=(lo, hi)),

            # Augmenter to transform images by moving pixels locally around using displacement fields.
            # Distort images locally by moving individual pixels around following a distortions field with
            # strength sigma_lo to sigma_hi. The strength of the movement is sampled per pixel from the range
            # alpha_lo to alpha_hi:
            "Elastic_Transformation": lambda alpha_lo, alpha_hi, sigma_lo, sigma_hi:
            iaa.ElasticTransformation(alpha=(alpha_lo, alpha_hi), sigma=(sigma_lo, sigma_hi)),

            # Augmenter to draw clouds in images.
            "Clouds": iaa.Clouds(),

            # Augmenter to draw fog in images.
            "Fog": iaa.Fog(),

            # Augmenter to add falling snowflakes to images.
            "Snowflakes": iaa.Snowflakes(),

            # Augmenter that calls a custom (lambda) function for each batch of input image.
            # All custom operations are defined in the Custom_Operations file (customFunc1 is placeholder)
            'Custom1': iaa.Lambda(CustomOperation.customFunc1, CustomOperation.keypoint_func)

        }

    # This function is used to add a single augmentation to the pipeline
    # The augmentation added is a combination of the ones found in augmentations
    # augmentations is a list of numbers that should correspond to the augmentations you want to combine
    def add_augmentation(self, augmentation_name, **kwargs):

        self.augs.append(self.Augmentations[augmentation_name](**kwargs))

    # This function returns a new list of augmented images based on the pipeline you create
    def augment_images(self):
        if self.augs:

            seq = iaa.Sequential(self.augs)
            images_aug = seq.augment_images(self.images)
            self.labels = self.labels  # labels should be identical

            return images_aug, self.labels

        else:
            raise TypeError("Augmentation pipe is currently empty")
