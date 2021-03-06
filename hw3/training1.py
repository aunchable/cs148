from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import Callback, History, ModelCheckpoint
from keras import backend as K
import os
import random
from PIL import Image
import numpy as np

random.seed(42)


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

#baseFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images'
#imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
#labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
#splitListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train_test_split.txt'
#bboxListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'


baseFolder = './CUB_200_2011/CUB_200_2011/images'
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
    image.thumbnail((width, height), Image.ANTIALIAS)
    background = Image.new('RGB', (width, height), (0, 0, 0))
    background.paste(
        image, (int((width - image.size[0]) / 2), int((height - image.size[1]) / 2))
    )
    return background


def imgGenerator(batchSize, train):
    """
    Yield X and Y data when the batch is filled.
    """
    X = np.zeros(shape=(batchSize, height, width, 3))
    Y = np.zeros(shape=(batchSize, 200))
    while True:
        if train:
            indices_for_batch = random.sample(indices_for_train, batchSize)
        else:
            indices_for_batch = random.sample(indices_for_val, batchSize)
        for i in range(batchSize):
            imgInfo = imageInfo[str(indices_for_batch[i])]
            X[i] = processImage(os.path.join(baseFolder, imgInfo[0]), imgInfo[2])
            imgOut = np.zeros(200)
            imgOut[int(imgInfo[1]) - 1] = 1
            Y[i] = imgOut
        yield X, Y


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(width, height, 3))

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
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit_generator(imgGenerator(32, True),
                              steps_per_epoch=int(float(len(indices_for_train)) / 32.0),
                              epochs=10,
                              validation_data=imgGenerator(100, False),
                              validation_steps=50)

print(history.history)

model.save('model1.h5')
