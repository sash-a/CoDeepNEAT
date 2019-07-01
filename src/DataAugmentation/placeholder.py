import Augmentor
import torchvision


def augment_batch(images, labels):

    #print(images[0])

    pipe = Augmentor.Pipeline()
    pipe.flip_left_right(1)
    pipe.rotate90(1)

    transforms = torchvision.transforms.Compose([
        pipe.torch_transform(),
        torchvision.transforms.ToTensor])

    return None, None
