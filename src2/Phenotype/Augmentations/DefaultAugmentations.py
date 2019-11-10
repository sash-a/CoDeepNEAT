import imgaug.augmenters as iaa

augmentations = [
    iaa.Fliplr(1),
    iaa.Affine(rotate=(-30, 30)),
    iaa.Affine(translate_px={"x": (-4, 4), "y": (-4, 4)}),
    iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}),
    iaa.Pad(px=(1, 4), keep_size=True, sample_independently=False),
    iaa.Crop(px=(1, 4), keep_size=True, sample_independently=False),
    iaa.CoarseDropout((0.05, 0.2), size_percent=(0.025, 0.5), per_channel=0.6),
    iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(0, iaa.Add((20, 50)))),
    iaa.Grayscale(alpha=(0.35, 0.75)),
    iaa.Noop()
]
