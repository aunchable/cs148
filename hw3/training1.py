from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import os
import random
import Image

height = 300
width = 300

baseFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images'
imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
imageInfo = {}

with f = open(imageListFile, 'r'):
    for line in f:
        [img_id, path] = line.split(' ')
        imageInfo[img_id] = [path]

with f = open(labelListFile, 'r'):
    for line in f:
        [img_id, label] = line.split(' ')
        imageInfo[img_id] = imageInfo[img_id].append(str(int(label) - 1))


def processImage(imagePath):
    image = Image.open(imagePath)
    image.thumbnail((width, height), Image.ANTIALIAS)
    background = Image.new('RGB', (width, height), (255, 255, 255))
    background.paste(
        image, (int((width - image.size[0]) / 2), int((height - image.size[1]) / 2))
    )
    return background


def imgGenerator(batchSize):
    """
    Yield X and Y data when the batch is filled.
    """

    X = np.zeros((batchSize, height, width, 3))
    Y = np.zeros((batchSize, 1))

    while True:
        indices_for_batch = random.sample(range(1, 11789), batchSize)
        for i in range(batchSize):
            imgInfo = imageInfo[str(indices_for_batch[i])]
            X[i] = processImage(os.path.join(baseFolder, imgInfo[0]))
            Y[i] = int(imgInfo[1])
        yield X, Y


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(width, height, 3)))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(imgGenerator(10), samples_per_epoch=10000, epochs=10)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

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
model.fit_generator(imgGenerator(10), samples_per_epoch=10000, epochs=10)
