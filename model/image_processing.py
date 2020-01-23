import numpy as np
from PIL import Image, ImageOps
import random

def PreprocessImage(im):
    mean = (104.00698793, 116.66876762, 122.67891434)
    if im.shape.__len__() < 3:
        im = np.stack((im,)*3, axis=-1)
    if im.shape[2] > 3:
        im = im[:, :, 0:3]
    im = im[:, :, ::-1]
    im -= mean
    try:
        im = im.transpose((2, 0, 1))
    except:
        print(im.shape)
    return im

def Mirror(im):
    return ImageOps.mirror(im)

def RandomCrop(im, output_size):
    width, height = im.size
    left = random.randint(0, width - output_size - 1)
    top = random.randint(0, height - output_size - 1)
    im = im.crop((left, top, left + output_size, top + output_size))
    return im

def CenterCrop(im, crop_size, output_size):
    width, height = im.size
    if width != crop_size:
        left = (width - crop_size) / 2
        right = width - left
        im = im.crop((left, 0, right, height))
    if height != crop_size:
        top = (height - crop_size) / 2
        bot = height - top
        im = im.crop((0, top, width, bot))
    return im.resize((output_size, output_size), Image.ANTIALIAS)
