from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
from keras import backend as K
import os
import random
from PIL import Image
import numpy as np

random.seed(42)

height = 500
width = 500

baseFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images'
imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
imageInfo = {}

indices_for_train = random.sample(range(1, 11789), int(0.8 * 11788))
indices_for_val = list(set(range(1,11789)) - set(indices_for_train))

f = open(imageListFile, 'r')
for line in f:
    [img_id, path] = line.split(' ')
    imageInfo[img_id] = path[:-1]

f = open(labelListFile, 'r')
for line in f:
    [img_id, label] = line.split(' ')
    imageInfo[img_id] = [imageInfo[img_id], label]

def processImage(imagePath):
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
            X[i] = processImage(os.path.join(baseFolder, imgInfo[0]))
            imgOut = np.zeros(200)
            imgOut[int(imgInfo[1]) - 1] = 1
            Y[i] = imgOut
        yield X, Y


# create the base pre-trained model
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(width, height, 3))
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(width, height, 3))

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

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# for i, layer in enumerate(model.layers):
#    print(i, layer.name)

# train the model on the new data for a few epochs
history1 = model.fit_generator(imgGenerator(10, True),
                               steps_per_epoch=int(0.8 * 11788 / 10.0),
                               epochs=10,
                               validation_data=imgGenerator(100, False),
                               validation_steps=20)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(model.layers):
#    print(i, layer.name)

# we chose to train everything but the top 2 inception blocks, i.e. we will
# freeze the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history2 = model.fit_generator(imgGenerator(10, True),
                               steps_per_epoch=int(0.8 * 11788 / 10.0),
                               epochs=10,
                               validation_data=imgGenerator(100, False),
                               validation_steps=20)
