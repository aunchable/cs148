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
    name_label[path[:-1].split('/')[-1]] = img_id
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


# create the base pre-trained model
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(width, height, 3))
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(height, width, 3))

# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model = load_model('/Users/anshulramachandran/Desktop/model1.h5')


# create the base pre-trained model
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(width, height, 3))
base_model2 = InceptionV3(weights='imagenet', include_top=False, input_shape=(height, width, 3))

# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# add a global spatial average pooling layer
x2 = base_model2.output
x2 = GlobalAveragePooling2D()(x2)
# let's add a fully-connected layer
predictions2 = Dense(1024, activation='relu')(x2)

# this is the model we will train
model2 = Model(inputs=base_model2.input, outputs=predictions2)

for i, layer in enumerate(model2.layers):
    print(i, layer.name, model.layers[i].name)
    layer.set_weights(model.layers[i].get_weights())


inters = []

count = 0
for path, subdirs, files in os.walk(validationFolder):
    for name in files:
        if name[0] != '.':
            img_id = float(name_label[name])
            curr_img = np.asarray([np.asarray(processImage(os.path.join(path, name)))])
            inter_output = np.insert(model2.predict(curr_img/255.0)[0], 0, img_id)
            inters.append(inter_output)
            count += 1
            if count % 100 == 0:
                print(count)

with open("./intermediates/validation1.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(inters)