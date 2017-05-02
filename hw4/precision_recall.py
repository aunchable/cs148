from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.callbacks import Callback, History, ModelCheckpoint
from keras.layers.core import Reshape
from keras.layers.merge import Concatenate
from keras import backend as K

# Our numerical workhorses
import numpy as np

# Scikit-image submodules
import skimage.filters
import skimage.io
import skimage.morphology
from skimage import measure

import matplotlib.pyplot as plt

import scipy.io as io
from PIL import Image

import os



trainFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train'
validationFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation'
imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
splitListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train_test_split.txt'
bboxListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'


# trainFolder = './CUB_200_2011/CUB_200_2011/train3'
# validationFolder = './CUB_200_2011/CUB_200_2011/validation3'
# imageListFile = './CUB_200_2011/CUB_200_2011/images.txt'
# labelListFile = './CUB_200_2011/CUB_200_2011/image_class_labels.txt'
# splitListFile = './CUB_200_2011/CUB_200_2011/train_test_split.txt'
# bboxListFile = './CUB_200_2011/CUB_200_2011/bounding_boxes.txt'

imageInfo = {}
name_label = {}

indices_for_train = []
indices_for_val = []

f = open(splitListFile, 'r')
for line in f:
    [img_id, img_class] = line.split(' ')
    if img_class[0] == '1':
        indices_for_train.append(img_id)
    else:
        indices_for_val.append(img_id)

f = open(labelListFile, 'r')
for line in f:
    [img_id, label] = line.split(' ')
    imageInfo[img_id] = label[:-1]

f = open(imageListFile, 'r')
for line in f:
    [img_id, path] = line.split(' ')
    name_label[path[:-1].split('/')[-1]] = int(imageInfo[img_id]) - 1
    imageInfo[img_id] = [path[:-1], imageInfo[img_id]]

f = open(bboxListFile, 'r')
for line in f:
    [img_id, x1, y1, w, h] = line.split(' ')
    x1 = float(x1)
    y1 = float(y1)
    x2 = x1 + float(w)
    y2 = y1 + float(h)
    imageInfo[img_id] = [imageInfo[img_id][0], imageInfo[img_id][1], [x1, y1, x2, y2]]

def processImage(imagePath):
    image = Image.open(imagePath)
    (currw, currh) = image.size
    if currw >= currh:
        image = image.resize((width, int(float(currh) * float(width) / float(currw))))
    else:
        image = image.resize((int(float(currw) * float(height) / float(currh)), height))
    background = Image.new('RGB', (width, height), (0, 0, 0))
    background.paste(
        image, (int((width - image.size[0]) / 2), int((height - image.size[1]) / 2))
    )
    return background


# Computes the intersection over union parameter
def intersection_over_union(boxA, boxB):
	xA = max(boxA[1], boxB[1])
	yA = max(boxA[0], boxB[0])
	xB = min(boxA[3], boxB[3])
	yB = min(boxA[2], boxB[2])

	interArea = (xB - xA + 1) * (yB - yA + 1)

	boxAArea = (boxA[3] - boxA[1] + 1) * (boxA[2] - boxA[0] + 1)
	boxBArea = (boxB[3] - boxB[1] + 1) * (boxB[2] - boxB[0] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

# Checks if two boxes overlap
def check_overlap(box1, box2):
    horiz = (box1[1] <= box2[3]) and (box1[3] >= box2[1])
    vert = (box1[0] <= box2[2]) and (box1[2] >= box2[0])
    return (horiz and vert)
