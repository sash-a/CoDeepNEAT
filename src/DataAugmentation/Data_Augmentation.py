# The Augmentor library is integral to this class (must cite)
import Augmentor

# The AugmenationScheme class allows for the creation of a pipeline consiting of different augmentations.
# Once all the desired augmentations are chosen for the pipeline.
# The pipeline can be used to create a DA scheme for a given image set

class AugmentationScheme:

    # Upon initialisation we create our pipeline which will eventually become our DA scheme
    def __init__(self, imagePath):
        self.imagePath = imagePath
        self.pipe = Augmentor.Pipeline(imagePath)

    # Every function requires you to specify a probability, which is used to decide if an operation is applied
    # to an image as it is passed through the augmentation pipeline.

    # Performs Histogram Equalisation on images
    def histrogramEqualisation(self, probability):
        self.pipe.histogram_equalisation(probability)

    # Converts images into greyscale (images will only have shades of grey)
    def greyScale(self, probability):
        self.pipe.greyscale(probability)

    # Negates images (reverses pixel values)
    def invert(self, probability):
        self.pipe.invert(probability)

    # Converts images into black and white, (1-bit, monochrome colour palette)
    def blackAndWhite(self, probability, threshold):
        self.pipe.black_and_white(probability, threshold)

    # Randomly changes image's brightness
    def randomBrightness(self, probability, min_factor, max_factor):
        self.pipe.random_brightness(probability,min_factor, max_factor)

    # Randomly changes image's colour saturation
    def randomColor(self, probability, min_factor, max_factor):
        self.pipe.random_color(probability, min_factor, max_factor)

    # Randomly changes image's contrast
    def randomContrast(self, probability, min_factor, max_factor):
        self.pipe.random_contrast(probability, min_factor, max_factor)

    # Performs perspective skewing on images
    def skew(self, probability, magnitude, skew_type):

        # magnitude: degree to which the skew is performed

        # skew_type:
        # "RANDOM" = 0
        # "TILT" = 1
        # "TILT_TOP_BOTTOM" = 2
        # "TILT_LEFT_RIGHT" = 3
        # "CORNER" = 4

        if(skew_type == 0):
            self.pipe.skew(probability, magnitude)
        elif(skew_type == 1):
            self.pipe.skew_tilt(probability, magnitude)
        elif(skew_type == 2):
            self.pipe.skew_top_bottom(probability, magnitude)
        elif(skew_type == 3):
            self.pipe.skew_left_right(probability, magnitude)
        elif(skew_type == 4):
            self.pipe.skew_corner(probability, magnitude)
        else:
            print("Error: non-existent skew_type given")

    # Performs rotation without automatically cropping the image
    def rotateStandard(self, probability, max_left_rotation, max_right_rotation, expand=False):
        self.pipe.rotate_without_crop(probability, max_left_rotation, max_right_rotation, expand)

    # Performs rotations in multiples of 90 degrees
    def rotateHard(self, probability, rotation):

        # rotation:
        # 90: rotate image 90 degrees
        # 180: rotate image 180 degrees
        # 270: rotate image 270 degrees
        # -1: rotate image randomly by either 90, 180 or 270 degrees

        if(rotation == 90):
            self.pipe.rotate90(probability)
        elif(rotation == 180):
            self.pipe.rotate180(probability)
        elif(rotation == 270):
            self.pipe.rotate270(probability)
        elif(rotation == -1):
            self.pipe.rotate_random_90(probability)
        else:
            print("Error: non-existent rotation given")

    # Perofrms rotatations on images by arbitrary numbers of degrees
    def rotateRange(self, proability, max_left_rotation, max_right_rotation):
        self.pipe.rotate(proability, max_left_rotation, max_right_rotation)

    # Resizes images
    def resize(self, probability, width, height, resample_filter):

        # width: width in pixels to resize image to
        # height: height in pixels to resize image to

        # resample_filter:
        # "NEAREST" = 0
        # "BICUBIC" = 1
        # "ANTIALIAS" = 2
        # "BILINEAR" = 3

        if(resample_filter == 0):
            self.pipe.resize(probability, width, height, "NEAREST")
        elif(resample_filter == 1):
            self.pipe.resize(probability, width, height, "BICUBIC")
        elif(resample_filter == 2):
            self.pipe.resize(probability, width, height, "ANTIALIAS")
        elif(resample_filter == 3):
            self.pipe.resize(probability, width, height, "BILINEAR")
        else:
            print("Error: non-existent resample_filter given")

    # Mirrors images through the x or y axis
    def flip(self, probability, direction):

        # direction:
        # "LEFT_RIGHT" = 0
        # "TOP_BOTTOM" = 1
        # "RANDOM" = 2

        if(direction == 0):
            self.pipe.flip_left_right(probability)
        elif(direction == 1):
            self.pipe.flip_top_bottom(probability)
        elif(direction == 2):
            self.pipe.flip_random(probability)
        else:
            print("Error: non-existent direction given")

    # Crops images based on specified size
    def crop(self, probability, width, height, centre):

        # width: the width in pixels of the area to crop from the image
        # height: the height in pixels of the area to crop from the image
        # centre: whether to crop from the centre of the image or a random location within the image (Boolean)

        self.pipe.crop_by_size(probability, width, height, centre)

    # Crops images based on specified percentage of image
    def cropPercentage(self, probability, percentage_area, centre):

        # percentage_area: the percentage area of the original image to crop (i.e. 0.5 = 50% of the image area)
        # centre: whether to crop from the centre of the image or a random location within the image

        if (centre == True):
            self.pipe.crop_centre(probability, percentage_area)
        else:
            self.pipe.crop_random(probability,percentage_area)

    # Shears images: tilts them in a certain direction (along x or y axis)
    def shear(self, probability, max_shear_left, max_shear_right):
        self.pipe.shear(probability, max_shear_left, max_shear_right)

    # Increases or decreases image in size by a certain factor (maintains aspect ratio)
    def scale(self, probability, scale_factor):

        # scale_factor: factor by which to scale i.e factor of 1.5 would scale image up by 150%
        self.pipe.scale(probability,scale_factor)

    # Performs randomised, elastic distortions on images
    def distort(self, probability, grid_width, grid_height, magnitude):

        # grid_width: width of the grid overlay
        # grid_height: height of the grid overlay
        # magnitude: controls the degree to which each distortion is applied to the overlaying distortion grid

        self.pipe.random_distortion(probability, grid_width, grid_height, magnitude)

    # Performs randomised, elastic gaussian distortions on images
    def gaussianDistortion(self, probability, grid_width, grid_height, magnitude, corner, method):

        # grid_width: width of the grid overlay
        # grid_height: height of the grid overlay
        # magnitude: controls the degree to which each distortion is applied to the overlaying distortion grid
        # corner: which corner of a picture to distort
            # "bell" = circular surface to distort
            # "ul" = upper left
            # "ur" = upper right
            # "dl" = down left
            # "dr" = down right
        # method:
        # "in" = apply max magnitude to chosen corner
        # "out" = inverse of "in"

        # Note: Function has 4 additional parameters (mex, mey, sdx, sdy) which are documented as being
        # used to generate 3D surfaces for similar distortions (surfaces based on normal distribution)
        # However, all of them have default values and because I don't really know what they do, have not
        # coded the functionality to change them (may easily do so if required)

        self.pipe.gaussian_distortion(probability, grid_width, grid_height, magnitude, corner, method)

    # Enlarges images (zooms) but returns a cropped region of zoomed image (of the same size as original image)
    def zoom(self, probability, min_factor, max_factor):
        self.pipe.zoom(probability, min_factor, max_factor)

    # Zooms into random areas of the image
    def zoomRandom(self, probability, percentage_area, randomise):

        # percentage_area: value between 0.1 and 1 that represents the area that will be cropped (0.1 = 10%)
        # randomise: if True, uses the percentage area as an upper bound and randomises the zoom level from 0.1 to
        # percentage_area
        self.pipe.zoom_random(probability, percentage_area, randomise)

    # Randomly selects a rectangle region in an image and erases its pixels with random values
    def randomErasing(self, probability, rectangle_area):

        # rectangle_area: percentage of the image to occlude

        self.pipe.random_erasing(probability, rectangle_area)

    # NOTE: the DA library provides functionality to create a custom image operation that can be applied to the pipeline

    def generateImages(self, numAugImages):
        self.pipe.sample(numAugImages)


aug = AugmentationScheme("/home/liron/PycharmProjects/DataAugmentation/Sample_Images")

#p = Augmentor.Pipeline("/home/liron/PycharmProjects/DataAugmentation/Sample_Images")
#p.gaussian_distortion(1, 20, 20, 10, "ul", "in")
#p.sample(5)
