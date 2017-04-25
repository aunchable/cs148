import os
import random
from PIL import Image
import numpy as np
import shutil
import scipy.misc

# trainFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train'
# validationFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation'
# imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
# labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
# splitListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train_test_split.txt'
# bboxListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'


trainFolder = './CUB_200_2011/train'
validationFolder = './CUB_200_2011/validation'
imageListFile = './CUB_200_2011/images.txt'
labelListFile = './CUB_200_2011/image_class_labels.txt'
splitListFile = './CUB_200_2011/train_test_split.txt'
bboxListFile = './CUB_200_2011/bounding_boxes.txt'

height = 256
width = 256

imageInfo = {}

paths_for_train = []
paths_for_val = []

f = open(imageListFile, 'r')
for line in f:
    [img_id, path] = line.split(' ')
    imageInfo[img_id] = path[:-1].split('/')[-1]

f = open(bboxListFile, 'r')
for line in f:
    [img_id, x1, y1, w, h] = line.split(' ')
    x1 = float(x1)
    y1 = float(y1)
    x2 = x1 + float(w)
    y2 = y1 + float(h)
    imageInfo[imageInfo[img_id]] = (x1, y1, x2, y2)
    imageInfo[img_id] = [imageInfo[img_id], (x1, y1, x2, y2)]

# print(imageInfo['Black_Footed_Albatross_0046_18.jpg'])

def processImage(imagePath, bbox):
    image = Image.open(imagePath)
    image = image.crop(bbox)
    return image
    # (currw, currh) = image.size
    # if currw >= currh:
    #     image = image.resize((width, int(float(currh) * float(width) / float(currw))))
    # else:
    #     image = image.resize((int(float(currw) * float(height) / float(currh)), height))
    # # image.thumbnail((width, height), Image.ANTIALIAS)
    # # return image
    # background = Image.new('RGB', (width, height), (0, 0, 0))
    # background.paste(
    #     image, (int((width - image.size[0]) / 2), int((height - image.size[1]) / 2))
    # )
    # return background

shutil.copytree(validationFolder, './CUB_200_2011/train2')
shutil.copytree(validationFolder, './CUB_200_2011/validation2')

for path, subdirs, files in os.walk('./CUB_200_2011/train2'):
    for name in files:
        if name[0] != '.':
            new_img = processImage(os.path.join(path, name), imageInfo[name])
            scipy.misc.imsave(os.path.join(path, name), np.asarray(new_img))

for path, subdirs, files in os.walk('./CUB_200_2011/validation2'):
    for name in files:
        if name[0] != '.':
            new_img = processImage(os.path.join(path, name), imageInfo[name])
            scipy.misc.imsave(os.path.join(path, name), np.asarray(new_img))
