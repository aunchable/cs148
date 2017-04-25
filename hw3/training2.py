from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import Callback, History, ModelCheckpoint
from keras import backend as K
import os
import random
from PIL import Image
import numpy as np

random.seed(42)


height = 256
width = 256

# trainFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train'
# validationFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation'
# imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
# labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
# splitListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train_test_split.txt'
# bboxListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'


trainFolder = './CUB_200_2011/CUB_200_2011/train'
validationFolder = './CUB_200_2011/CUB_200_2011/validation'
imageListFile = './CUB_200_2011/CUB_200_2011/images.txt'
labelListFile = './CUB_200_2011/CUB_200_2011/image_class_labels.txt'
splitListFile = './CUB_200_2011/CUB_200_2011/train_test_split.txt'
bboxListFile = './CUB_200_2011/CUB_200_2011/bounding_boxes.txt'

imageInfo = {}

indices_for_train = []
indices_for_val = []

f = open(splitListFile, 'r')
for line in f:
    [img_id, img_class] = line.split(' ')
    if img_class[0] == '1':
        indices_for_train.append(img_id)
    else:
        indices_for_val.append(img_id)

f = open(imageListFile, 'r')
for line in f:
    [img_id, path] = line.split(' ')
    imageInfo[img_id] = path[:-1]

f = open(labelListFile, 'r')
for line in f:
    [img_id, label] = line.split(' ')
    imageInfo[img_id] = [imageInfo[img_id], label[:-1]]

f = open(bboxListFile, 'r')
for line in f:
    [img_id, x1, y1, w, h] = line.split(' ')
    x1 = float(x1)
    y1 = float(y1)
    x2 = x1 + float(w)
    y2 = y1 + float(h)
    imageInfo[img_id] = [imageInfo[img_id][0], imageInfo[img_id][1], (x1, y1, x2, y2)]

def processImage(imagePath, bbox):
    image = Image.open(imagePath)
    #image = image.crop(bbox)
    image.thumbnail((width, height), Image.ANTIALIAS)
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
# and a logistic layer -- we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# we chose to train everything but the top 2 inception blocks, i.e. we will
# freeze the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

train_datagen = ImageDataGenerator(
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        trainFolder,
        target_size=(height, width),
        batch_size=32)

validation_generator = test_datagen.flow_from_directory(
        validationFolder,
        target_size=(height, width),
        batch_size=32)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit_generator(train_generator,
                              steps_per_epoch=128,
                              epochs=50,
                              validation_data=validation_generator,
                              validation_steps=int(float(len(indices_for_val)) / 32.0))

print(history.history)

model.save('model1.h5')
