import imgaug.augmenters as iaa
from src.DataAugmentation.CustomOperations import CustomOperation as CO

# Dictionary containing all possible augmentation functions
Augmentations = {

    # Convert images to HSV, then increase each pixel's Hue (H), Saturation (S) or Value/lightness (V) [0, 1, 2]
    # value by an amount in between lo and hi:
    "HSV": lambda channel, lo, hi: iaa.WithColorspace
    (to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(channel, iaa.Add((lo, hi)))),

    # The augmenter first transforms images to HSV color space, then adds random values (lo to hi)
    # to the H and S channels and afterwards converts back to RGB.
    # (independently per channel and the same value for all pixels within that channel)
    "Add_To_Hue_And_Saturation": lambda lo, hi: iaa.AddToHueAndSaturation((lo, hi), per_channel=True),

    # Increase each pixel’s channel-value (redness/greenness/blueness) [0, 1, 2] by value in between lo and hi:
    "Increase_Channel": lambda channel, lo, hi: iaa.WithChannels(channel, iaa.Add((lo, hi))),
    # Rotate each image’s channel [R=0, G=1, B=2] by value in between lo and hi degrees:
    "Rotate_Channel": lambda channel, lo, hi: iaa.WithChannels(channel, iaa.Affine(rotate=(lo, hi))),

    # Augmenter that never changes input images (“no operation”).
    "No_Operation": iaa.Noop(),

    # Pads images, i.e. adds columns/rows to them. Pads image by value in between lo and hi
    # percent relative to its original size (only accepts positive values in range[0, 1]):
    # If s_i is false, The value will be sampled once per image and used for all sides
    # (i.e. all sides gain/lose the same number of rows/columns)
    # NOTE: automatically resizes images back to their original size after it has augmented them.
    "Pad_Percent": lambda lo, hi, s_i: iaa.Pad(percent=(lo, hi), keep_size=True, sample_independently=s_i),

    # Pads images by a number of pixels between lo and hi
    # If s_i is false, The value will be sampled once per image and used for all sides
    # (i.e. all sides gain/lose the same number of rows/columns)
    "Pad_Pixels": lambda lo, hi, s_i: iaa.Pad(px=(lo, hi), keep_size=True, sample_independently=s_i),

    # Crops/cuts away pixels at the sides of the image.
    # Crops images by value in between lo and hi (only accepts positive values in range[0, 1]):
    # If s_i is false, The value will be sampled once per image and used for all sides
    # (i.e. all sides gain/lose the same number of rows/columns)
    # NOTE: automatically resizes images back to their original size after it has augmented them.
    "Crop_Percent": lambda lo, hi, s_i: iaa.Crop(percent=(lo, hi), keep_size=True, sample_independently=s_i),

    # Crops images by a number of pixels between lo and hi
    # If s_i is false, The value will be sampled once per image and used for all sides
    # (i.e. all sides gain/lose the same number of rows/columns)
    "Crop_Pixels": lambda lo, hi, s_i: iaa.Crop(px=(lo, hi), keep_size=True, sample_independently=s_i),

    # Flip/mirror percent (i.e 0.5) of the input images horizontally
    # The default probability is 0, so to flip all images, percent=1
    "Flip_lr": iaa.Fliplr(1),

    # Flip/mirror percent (i.e 0.5) of the input images vertically
    # The default probability is 0, so to flip all images, percent=1
    "Flip_ud": iaa.Flipud(1),

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
    # images (sampled once for all channels) and sampled three (RGB) times (channel-wise)
    # for the rest from the same normal distribution:
    "Additive_Gaussian_Noise": lambda lo, hi, percent:
    iaa.AdditiveGaussianNoise(scale=(lo, hi), per_channel=percent),

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
    "Coarse_Dropout": lambda d_lo, d_hi, s_lo, s_hi, percent:
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
    "Translate_Percent": lambda x_lo, x_hi, y_lo, y_hi:
    iaa.Affine(translate_percent={"x": (x_lo, x_hi), "y": (y_lo, y_hi)}),

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

    # Weather augmenters are computationally expensive and will not work effectively on certain data sets

    # Augmenter to draw clouds in images.
    "Clouds": iaa.Clouds(),

    # Augmenter to draw fog in images.
    "Fog": iaa.Fog(),

    # Augmenter to add falling snowflakes to images.
    "Snowflakes": iaa.Snowflakes(),

    # Replaces percent of all pixels in an image by either x or y
    "Replace_Element_Wise": lambda percent, x, y: iaa.ReplaceElementwise(percent, [x, y]),

    # Adds laplace noise (somewhere between gaussian and salt and peeper noise) to an image, sampled once per pixel
    # from a laplace distribution Laplace(0, s), where s is sampled per image and varies between lo and hi*255 for
    # percent of all images (sampled once for all channels) and sampled three (RGB) times (channel-wise)
    # for the rest from the same laplace distribution:
    "Additive_Laplace_Noise": lambda lo, hi, percent:
    iaa.AdditiveLaplaceNoise(scale=(lo, hi), per_channel=percent),

    # Adds poisson noise (similar to gaussian but different distribution) to an image, sampled once per pixel from
    # a poisson distribution Poisson(s), where s is sampled per image and varies between lo and hi for percent of
    # all images (sampled once for all channels) and sampled three (RGB) times (channel-wise)
    # for the rest from the same poisson distribution:
    "Additive_Poisson_Noise": lambda lo, hi, percent:
    iaa.AdditivePoissonNoise(lam=(lo, hi), per_channel=percent),

    # Adds salt and pepper noise to an image, i.e. some white-ish and black-ish pixels.
    # Replaces percent of all pixels with salt and pepper noise
    "Salt_And_Pepper": lambda percent: iaa.SaltAndPepper(percent),

    # Adds coarse salt and pepper noise to image, i.e. rectangles that contain noisy white-ish and black-ish pixels
    # Replaces percent of all pixels with salt/pepper in an image that has lo to hi percent of the input image size,
    # then upscales the results to the input image size, leading to large rectangular areas being replaced.
    "Coarse_Salt_And_Pepper": lambda percent, lo, hi: iaa.CoarseSaltAndPepper(percent, size_percent=(lo, hi)),

    # Adds salt noise to an image, i.e white-ish pixels
    # Replaces percent of all pixels with salt noise
    "Salt": lambda percent: iaa.Salt(percent),

    # Adds coarse salt noise to image, i.e. rectangles that contain noisy white-ish pixels
    # Replaces percent of all pixels with salt in an image that has lo to hi percent of the input image size,
    # then upscales the results to the input image size, leading to large rectangular areas being replaced.
    "Coarse_Salt": lambda percent, lo, hi: iaa.CoarseSalt(percent, size_percent=(lo, hi)),

    # Adds Pepper noise to an image, i.e Black-ish pixels
    # Replaces percent of all pixels with Pepper noise
    "Pepper": lambda percent: iaa.Pepper(percent),

    # Adds coarse pepper noise to image, i.e. rectangles that contain noisy black-ish pixels
    # Replaces percent of all pixels with salt in an image that has lo to hi percent of the input image size,
    # then upscales the results to the input image size, leading to large rectangular areas being replaced.
    "Coarse_Pepper": lambda percent, lo, hi: iaa.CoarsePepper(percent, size_percent=(lo, hi)),

    # In an alpha blending, two images are naively mixed. E.g. Let A be the foreground image, B be the background
    # image and a is the alpha value. Each pixel intensity is then computed as a * A_ij + (1-a) * B_ij.
    # Images passed in must be a numpy array of type (height, width, channel)
    "Blend_Alpha": lambda image_fg, image_bg, alpha: iaa.blend_alpha(image_fg, image_bg, alpha),

    # Blur/Denoise an image using a bilateral filter.
    # Bilateral filters blur homogeneous and textured areas, while trying to preserve edges.
    # Blurs all images using a bilateral filter with max distance d_lo to d_hi with ranges for sigma_colour
    # and sigma space being define by sc_lo/sc_hi and ss_lo/ss_hi
    "Bilateral_Blur": lambda d_lo, d_hi, sc_lo, sc_hi, ss_lo, ss_hi:
    iaa.BilateralBlur(d=(d_lo, d_hi), sigma_color=(sc_lo, sc_hi), sigma_space=(ss_lo, ss_hi)),

    # Augmenter that sharpens images and overlays the result with the original image.
    # Create a motion blur augmenter with kernel size of (kernel x kernel) and a blur angle of either x or y degrees
    # (randomly picked per image).
    "Motion_Blur": lambda kernel, x, y: iaa.MotionBlur(k=kernel, angle=[x, y]),

    # Augmenter to apply standard histogram equalization to images (similar to CLAHE)
    "Histogram_Equalization": iaa.HistogramEqualization(),

    # Augmenter to perform standard histogram equalization on images, applied to all channels of each input image
    "All_Channels_Histogram_Equalization": iaa.AllChannelsHistogramEqualization(),

    # Contrast Limited Adaptive Histogram Equalization (CLAHE). This augmenter applies CLAHE to images, a form of
    # histogram equalization that normalizes within local image patches.
    # Creates a CLAHE augmenter with clip limit uniformly sampled from [cl_lo..cl_hi], i.e. 1 is rather low contrast
    # and 50 is rather high contrast. Kernel sizes of SxS, where S is uniformly sampled from [t_lo..t_hi].
    # Sampling happens once per image. (Note: more parameters are available for further specification)
    "CLAHE": lambda cl_lo, cl_hi, t_lo, t_hi: iaa.CLAHE(clip_limit=(cl_lo, cl_hi), tile_grid_size_px=(t_lo, t_hi)),

    # Contrast Limited Adaptive Histogram Equalization (refer above), applied to all channels of the input images.
    # CLAHE performs histogram equalization within image patches, i.e. over local neighbourhoods
    "All_Channels_CLAHE": lambda cl_lo, cl_hi, t_lo, t_hi:
    iaa.AllChannelsCLAHE(clip_limit=(cl_lo, cl_hi), tile_grid_size_px=(t_lo, t_hi)),

    # Augmenter that changes the contrast of images using a unique formula (using gamma).
    # Multiplier for gamma function is between lo and hi,, sampled randomly per image (higher values darken image)
    # For percent of all images values are sampled independently per channel.
    "Gamma_Contrast": lambda lo, hi, percent: iaa.GammaContrast((lo, hi), per_channel=percent),

    # Augmenter that changes the contrast of images using a unique formula (linear).
    # Multiplier for linear function is between lo and hi, sampled randomly per image
    # For percent of all images values are sampled independently per channel.
    "Linear_Contrast": lambda lo, hi, percent: iaa.LinearContrast((lo, hi), per_channel=percent),

    # Augmenter that changes the contrast of images using a unique formula (using log).
    # Multiplier for log function is between lo and hi, sampled randomly per image.
    # For percent of all images values are sampled independently per channel.
    # Values around 1.0 lead to a contrast-adjusted images. Values above 1.0 quickly lead to partially broken
    # images due to exceeding the datatype’s value range.
    "Log_Contrast": lambda lo, hi, percent: iaa.LogContrast((lo, hi), per_channel=percent),

    # Augmenter that changes the contrast of images using a unique formula (sigmoid).
    # Multiplier for sigmoid function is between lo and hi, sampled randomly per image. c_lo and c_hi decide the
    # cutoff value that shifts the sigmoid function in horizontal direction (Higher values mean that the switch
    # from dark to light pixels happens later, i.e. the pixels will remain darker).
    # For percent of all images values are sampled independently per channel:
    "Sigmoid_Contrast": lambda lo, hi, c_lo, c_hi, percent:
    iaa.SigmoidContrast((lo, hi), (c_lo, c_hi), per_channel=percent),

    # Augmenter that calls a custom (lambda) function for each batch of input image.
    # Extracts Canny Edges from images (refer to description in CO)
    # Good default values for min and max are 100 and 200
    'Custom_Canny_Edges': lambda min_val, max_val: iaa.Lambda(
        func_images=CO.Edges(min_value=min_val, max_value=max_val)),

}
