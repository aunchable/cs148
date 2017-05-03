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


height = 256
width = 256

trainFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train'
validationFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation3'
imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
splitListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train_test_split.txt'
bboxListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'


# trainFolder = './CUB_200_2011/CUB_200_2011/train'
# validationFolder = './CUB_200_2011/CUB_200_2011/validation'
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
    imageInfo[img_id] = [imageInfo[img_id][0], imageInfo[img_id][1], (x1, y1, x2, y2)]

def processImage(imagePath):
    image = Image.open(imagePath)
    (currw, currh) = image.size
    if currw >= currh:
        image = image.resize((width, int(float(currh) * float(width) / float(currw))))
    else:
        image = image.resize((int(float(currw) * float(height) / float(currh)), height))
    # image.thumbnail((width, height), Image.ANTIALIAS)
    # return image
    background = Image.new('RGB', (width, height), (0, 0, 0))
    background.paste(
        image, (int((width - image.size[0]) / 2), int((height - image.size[1]) / 2))
    )

    return background

model = load_model('/Users/anshulramachandran/Desktop/model_mbcrop2.h5')

# Generate confusion matrix
confusion_matrix = np.zeros(shape=(200,200))

count = 0
for path, subdirs, files in os.walk(validationFolder):
    for name in files:
        if name[0] != '.':
            class_true = name_label[name]
            curr_img = np.asarray([np.asarray(processImage(os.path.join(path, name)))])
            class_pred = np.argmax(model.predict(curr_img/255.0))
            confusion_matrix[class_true][class_pred] += 1
            count += 1
            if count % 100 == 0:
                print(count)

confusion_matrix = confusion_matrix.astype(int)
confusion_matrix = -confusion_matrix
confusion_matrix_img = np.zeros(shape=(1000, 1000))
for i in range(len(confusion_matrix_img)):
    for j in range(len(confusion_matrix_img[0])):
        confusion_matrix_img[i][j] = confusion_matrix[int(i/5.0), int(j/5.0)]
scipy.misc.imsave('./graphs/confusion_matrix2.jpg', confusion_matrix_img)
for i in range(200):
    confusion_matrix[i][i] = 0
confusion_matrix_img = np.zeros(shape=(1000, 1000))
for i in range(len(confusion_matrix_img)):
    for j in range(len(confusion_matrix_img[0])):
        confusion_matrix_img[i][j] = confusion_matrix[int(i/5.0), int(j/5.0)]
scipy.misc.imsave('./graphs/confusion_matrix2_no_diagonal.jpg', confusion_matrix_img)
