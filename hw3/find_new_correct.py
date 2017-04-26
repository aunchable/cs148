from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import Callback, History, ModelCheckpoint
from keras import backend as K
import os
import random
from PIL import Image
import numpy as np
import scipy.misc
import csv

height = 256
width = 256

trainFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train'
validationFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation'
validationFolder2 = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation2'
imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
splitListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train_test_split.txt'
bboxListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'



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
    name_label[path[:-1].split('/')[-1]] = [img_id, int(imageInfo[img_id]) - 1]
    imageInfo[img_id] = [path[:-1], imageInfo[img_id]]

f = open(bboxListFile, 'r')
for line in f:
    [img_id, x1, y1, w, h] = line.split(' ')
    x1 = float(x1)
    y1 = float(y1)
    x2 = x1 + float(w)
    y2 = y1 + float(h)
    imageInfo[img_id] = [imageInfo[img_id][0], imageInfo[img_id][1], (x1, y1, x2, y2)]


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


model1 = load_model('/Users/anshulramachandran/Desktop/model2_2.h5')
model2 = load_model('/Users/anshulramachandran/Desktop/model3.h5')


count = 0
for path, subdirs, files in os.walk(validationFolder):
    for name in files:
        if name[0] != '.':
            img_id = name_label[name][0]
            true_class = name_label[name][1]
            subpath = os.path.join(path.split('/')[-1], name)

            curr_img1 = np.asarray([np.asarray(processImage(os.path.join(validationFolder, subpath)))])
            curr_img2 = np.asarray([np.asarray(processImage(os.path.join(validationFolder2, subpath)))])
            class1_pred = np.argmax(model1.predict(curr_img1/255.0))
            class2_pred = np.argmax(model2.predict(curr_img2/255.0))

            if class2_pred == true_class and class1_pred != true_class:
                print(img_id, name, class1_pred, class2_pred, true_class)
            count += 1
            if count % 100 == 0:
                print(count)
