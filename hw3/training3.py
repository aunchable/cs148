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

# we chose to train everything but the top 2 inception blocks, i.e. we will
# freeze the first 172 layers and unfreeze the rest:
for layer in base_model.layers:
    layer.trainable = False
# for layer in model.layers[:172]:
#    layer.trainable = False
# for layer in model.layers[172:]:
#    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
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
#
# history = model.fit_generator(train_generator,
#                               steps_per_epoch=5,
#                               epochs=2,
#                               validation_data=validation_generator,
#                               validation_steps=3,
#                               callbacks=[history2])
#
print(history.history)
print(history2.losses, history2.val_losses, history2.accs, history2.val_accs)

model.save('model2_2.h5')
#
# model = load_model('model2.h5')

# # Generate confusion matrix
# confusion_matrix = np.zeros(shape=(200,200))
#
# for path, subdirs, files in os.walk(validationFolder):
# # for path, subdirs, files in os.walk('./CUB_200_2011/CUB_200_2011/validation'):
#     for name in files:
#         if name[0] != '.':
#             class_true = name_label[name]
#             class_pred = np.argmax(model.predict(np.asarray([np.asarray(processImage(os.path.join(path, name)))])))
#             confusion_matrix[class_true][class_pred] += 1
#
# # for i in range(len(testY)):
# #     class_true = np.argmax(testY[i])
# #     digit_pred = np.argmax(predY[i])
# #     confusion_matrix[digit_true][digit_pred] += 1
# confusion_matrix = confusion_matrix.astype(int)
# confusion_matrix = -confusion_matrix
# confusion_matrix_img = np.zeros(shape=(1000, 1000))
# for i in range(len(confusion_matrix_img)):
#     for j in range(len(confusion_matrix_img[0])):
#         confusion_matrix_img[i][j] = confusion_matrix[int(i/5.0), int(j/5.0)]
# scipy.misc.imsave('confusion_matrix2.jpg', confusion_matrix_img)
# for i in range(10):
#     confusion_matrix[i][i] = 0
# confusion_matrix_img = np.zeros(shape=(1000, 1000))
# for i in range(len(confusion_matrix_img)):
#     for j in range(len(confusion_matrix_img[0])):
#         confusion_matrix_img[i][j] = confusion_matrix[int(i/5.0), int(j/5.0)]
# scipy.misc.imsave('confusion_matrix2_no_diagonal.jpg', confusion_matrix_img)
