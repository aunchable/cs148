# FOR PROBLEM 2

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

class TrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_acc'))


height = 256
width = 256

# trainFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train2'
# validationFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation2'
# imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
# labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
# splitListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train_test_split.txt'
# bboxListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'


trainFolder = './CUB_200_2011/train2'
validationFolder = './CUB_200_2011/validation2'
imageListFile = './CUB_200_2011/images.txt'
labelListFile = './CUB_200_2011/image_class_labels.txt'
splitListFile = './CUB_200_2011/train_test_split.txt'
bboxListFile = './CUB_200_2011/bounding_boxes.txt'

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
    image = image.crop(bbox)
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



# create the base pre-trained model
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

for layer in base_model.layers:
    layer.trainable = False

# we need to recompile the model for these modifications to take effect
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=15,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

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
history2 = TrainingHistory()
history = model.fit_generator(train_generator,
                              steps_per_epoch=int(float(len(indices_for_train)) / 32.0),
                              epochs=12,
                              validation_data=validation_generator,
                              validation_steps=int(float(len(indices_for_val)) / 32.0),
                              callbacks=[history2])

print(history.history)
print(history2.losses, history2.val_losses, history2.accs, history2.val_accs)

model.save('model2.h5')
