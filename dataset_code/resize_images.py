import urllib
from joblib import Parallel, delayed
from PIL import Image
import os

def resize(im, size):
    im = im.resize((size, size), Image.ANTIALIAS)
    return im

def resize_dataset(id):
    try:
        img = Image.open(dataset_root + 'img_align_celeba/' + id)
        img = resize(img, 224)
        image_path = dest_path + str(id) 
        img.save(image_path)
    except:
        print("ERROR")


dataset_root = '/media/ssd2/CelebA/'
dest_path = dataset_root + 'img_resized/'

img_names = []
for line in open(dataset_root + 'Eval/list_eval_partition.txt', 'r'):
    img_name = line.split(' ')[0]
    img_names.append(img_name)

print("Resizing")
Parallel(n_jobs=64)(delayed(resize_dataset)(id) for id in img_names)
print("DONE")