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
import csv

height = 299
width = 299

imgFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images'
trainFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train'
validationFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation'
imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
splitListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train_test_split.txt'
bboxListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'

# imgFolder = './CUB_200_2011/images'
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

def modify_bbox(x1, y1, x2, y2, height, width, imagePath):
    image = Image.open(imagePath)
    (currw, currh) = image.size
    if currw >= currh:
        resize = float(width) / float(currw)
    else:
        resize = float(height) / float(currh)
    x1 = resize * x1
    y1 = resize * y1
    x2 = resize * x2
    y2 = resize * y2
    if currw >= currh:
        y1 = y1 + 0.5 * (height - resize * currh)
        y2 = y2 + 0.5 * (height - resize * currh)
    else:
        x1 = x1 + 0.5 * (width - resize * currw)
        x2 = x2 + 0.5 * (width - resize * currw)
    return([x1, y1, x2, y2])


f = open(bboxListFile, 'r')
for line in f:
    [img_id, x1, y1, w, h] = line.split(' ')
    x1 = float(x1)
    y1 = float(y1)
    x2 = x1 + float(w)
    y2 = y1 + float(h)
    bbox = modify_bbox(x1, y1, x2, y2, height, width, os.path.join(imgFolder, imageInfo[img_id][0]))
    imageInfo[img_id] = [imageInfo[img_id][0], bbox]


f = open(imageListFile, 'r')
for line in f:
    [img_id, path] = line.split(' ')
    name_label[path[:-1].split('/')[-1]] = [imageInfo[img_id][0], imageInfo[img_id][1]]
    imageInfo[img_id] = [path[:-1], imageInfo[img_id][0], imageInfo[img_id][1]]

print(name_label)
assert(False)

aspect_ratios = [[1.0, 1.0], [1.0, 0.67], [0.67, 1.0], [0.8, 0.6], [0.6, 0.8], [1.0, 0.75], [0.75, 1.0], [1.0, 0.6], [0.6, 1.0], [1.0, 0.4], [0.4, 1.0]]

def make_prior_box(x1, y1, x2, y2, ar):
    return [x1 + 0.5*(1.0-ar[0])*(x2 - x1),
            y1 + 0.5*(1.0-ar[1])*(y2 - y1),
            x2 - 0.5*(1.0-ar[0])*(x2 - x1),
            y2 - 0.5*(1.0-ar[1])*(y2 - y1)]

def get_pboxes(height, width):

    pboxes = []

    for ar in aspect_ratios:
        for i in range(8):
            for j in range(8):
                pboxes.append(make_prior_box(i*height/9.0, j*width/9.0, (i+1)*height/9.0, (j+1)*width/9.0, ar))

    for ar in aspect_ratios:
        for i in range(6):
            for j in range(6):
                pboxes.append(make_prior_box(i*height/7.0, j*width/7.0, (i+1)*height/7.0, (j+1)*width/7.0, ar))

    for ar in aspect_ratios:
        for i in range(4):
            for j in range(4):
                pboxes.append(make_prior_box(i*height/5.0, j*width/5.0, (i+1)*height/5.0, (j+1)*width/5.0, ar))

    for ar in aspect_ratios:
        for i in range(3):
            for j in range(3):
                pboxes.append(make_prior_box(i*height/4.0, j*width/4.0, (i+1)*height/4.0, (j+1)*width/4.0, ar))

    for ar in aspect_ratios:
        for i in range(2):
            for j in range(2):
                pboxes.append(make_prior_box(i*height/3.0, j*width/3.0, (i+1)*height/3.0, (j+1)*width/3.0, ar))

    pboxes.append([0.1*height, 0.1*width, 0.9*height, 0.9*width])

    return np.asarray(pboxes)

pboxes = get_pboxes(height, width)


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
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

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


model = load_model('/Users/anshulramachandran/Desktop/multibox1.h5')

iou_scores = []

count = 0
for path, subdirs, files in os.walk(validationFolder):
    for name in files:
        if name[0] != '.':
            label = name_label[name][0]
            gbox = name_label[name][1]
            curr_img = np.asarray([np.asarray(processImage(os.path.join(path, name)))])
            pred_boxes = model.predict(curr_img/255.0)
            print(pred_boxes.shape)
            assert(0 == 1)
            pred_boxes[:, :4] = pred_boxes[:, :4] + pboxes
            for box_conf in pred_boxes:
                if check_overlap(box_conf[:4], gbox):
                    iou_score = intersection_over_union(box_conf[:4], gbox)
                else:
                    iou_score = 0.0
                iou_scores.append([iou_score, box_conf[4], label])
            print(iou_scores)
            assert(False)
            count += 1
            if count % 100 == 0:
                print(count)

iou_scores = np.asarray(iou_scores)
iou_scores[iou_scores[:,1].argsort()]
print(iou_scores[:100])
assert(False)

iou_scores = iou_scores[::-1]
print(iou_scores[:100])
assert(False)

with open('/Users/anshulramachandran/Desktop/ious.csv', 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile, delimiter=',')
    for i in range(len(iou_scores)):
        datawriter.writerow(iou_scores[i])
