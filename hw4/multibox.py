# FOR PROBLEM 3

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
import os
import random
from PIL import Image
import numpy as np
import scipy.misc

import tensorflow as tf

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



height = 299
width = 299

num_bboxes_per_cell = 11
alpha = 10

y_len = 1420

batchSize = 32

random.seed(42)

# imgFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images'
# trainFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train'
# validationFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation'
# imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'
# labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
# splitListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train_test_split.txt'
# bboxListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'

imgFolder = './CUB_200_2011/images'
trainFolder = './CUB_200_2011/train'
validationFolder = './CUB_200_2011/validation'
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
    imageInfo[img_id] = [imageInfo[img_id][0], imageInfo[img_id][1], bbox]

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


def imgGenerator(batch_size, train):
    """
    Yield X and Y data when the batch is filled.
    """
    X = np.zeros(shape=(batch_size, height, width, 3))
    Y = np.zeros(shape=(batch_size, y_len, 5))
    while True:
        if train:
            indices_for_batch = random.sample(indices_for_train, batch_size)
        else:
            indices_for_batch = random.sample(indices_for_val, batch_size)
        for i in range(batch_size):
            imgInfo = imageInfo[str(indices_for_batch[i])]
            if train:
                X[i] = np.asarray(processImage(os.path.join(trainFolder, imgInfo[0]))) / 255.0
            else:
                X[i] = np.asarray(processImage(os.path.join(validationFolder, imgInfo[0]))) /255.0
            imgOut = np.zeros(shape=(y_len, 5))
            for j in range(y_len):
                imgOut[j, 0:4] = imgInfo[2]
            Y[i] = imgOut
        yield X, Y


def bounding_box_prediction_layers(inputs, bboxes_per_cell, batch_size):

    endpoints = {}

    branch88 = Conv2D(96, (1, 1), strides=1, padding='same')(inputs)
    branch88 = Conv2D(96, (3, 3), strides=1, padding='same')(branch88)
    endpoints['branch88_locs'] = Conv2D(bboxes_per_cell*4, (1, 1), strides=1, padding='same')(branch88)
    endpoints['branch88_confs'] = Conv2D(bboxes_per_cell, (1, 1), strides=1, padding='same', activation='sigmoid')(branch88)

    branch66 = Conv2D(96, (3, 3), strides=1, padding='same')(inputs)
    branch66 = Conv2D(96, (3, 3), strides=1, padding='valid')(branch66)
    endpoints['branch66_locs'] = Conv2D(bboxes_per_cell*4, (1, 1), strides=1, padding='same')(branch66)
    endpoints['branch66_confs'] = Conv2D(bboxes_per_cell, (1, 1), strides=1, padding='same', activation='sigmoid')(branch66)

    rightBranchBase = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(inputs)

    branch44 = Conv2D(128, (3, 3), strides=1, padding='same')(rightBranchBase)
    endpoints['branch44_locs'] = Conv2D(bboxes_per_cell*4, (1, 1), strides=1, padding='same')(branch44)
    endpoints['branch44_confs'] = Conv2D(bboxes_per_cell, (1, 1), strides=1, padding='same', activation='sigmoid')(branch44)

    branch33 = Conv2D(128, (1, 1), strides=1, padding='same')(rightBranchBase)
    branch33 = Conv2D(96, (2, 2), strides=1, padding='valid')(branch33)
    endpoints['branch33_locs'] = Conv2D(bboxes_per_cell*4, (1, 1), strides=1, padding='same')(branch33)
    endpoints['branch33_confs'] = Conv2D(bboxes_per_cell, (1, 1), strides=1, padding='same', activation='sigmoid')(branch33)

    branch22 = Conv2D(128, (1, 1), strides=1, padding='same')(rightBranchBase)
    branch22 = Conv2D(96, (3, 3), strides=1, padding='valid')(branch22)
    endpoints['branch22_locs'] = Conv2D(bboxes_per_cell*4, (1, 1), strides=1, padding='same')(branch22)
    endpoints['branch22_confs'] = Conv2D(bboxes_per_cell, (1, 1), strides=1, padding='same', activation='sigmoid')(branch22)

    branch11 = AveragePooling2D((8, 8), strides=(1, 1), padding='valid')(inputs)
    endpoints['branch11_locs'] = Conv2D(4, (1, 1), strides=1, padding='same')(branch11)
    endpoints['branch11_confs'] = Conv2D(1, (1, 1), strides=1, padding='same', activation='sigmoid')(branch11)

    # batch_size = tf.shape(inputs)[0]

    locs88 = Reshape((-1,4))(endpoints['branch88_locs'])
    confs88 = Reshape((-1,1))(endpoints['branch88_confs'])
    locs66 = Reshape((-1,4))(endpoints['branch66_locs'])
    confs66 = Reshape((-1,1))(endpoints['branch66_confs'])
    locs44 = Reshape((-1,4))(endpoints['branch44_locs'])
    confs44 = Reshape((-1,1))(endpoints['branch44_confs'])
    locs33 = Reshape((-1,4))(endpoints['branch33_locs'])
    confs33 = Reshape((-1,1))(endpoints['branch33_confs'])
    locs22 = Reshape((-1,4))(endpoints['branch22_locs'])
    confs22 = Reshape((-1,1))(endpoints['branch22_confs'])
    locs11 = Reshape((-1,4))(endpoints['branch11_locs'])
    confs11 = Reshape((-1,1))(endpoints['branch11_confs'])

    # locs88 = tf.reshape(endpoints['branch88_locs'], [batch_size, -1])
    # confs88 = tf.reshape(endpoints['branch88_confs'], [batch_size, -1])
    # locs66 = tf.reshape(endpoints['branch66_locs'], [batch_size, -1])
    # confs66 = tf.reshape(endpoints['branch66_confs'], [batch_size, -1])
    # locs44 = tf.reshape(endpoints['branch44_locs'], [batch_size, -1])
    # confs44 = tf.reshape(endpoints['branch44_confs'], [batch_size, -1])
    # locs33 = tf.reshape(endpoints['branch33_locs'], [batch_size, -1])
    # confs33 = tf.reshape(endpoints['branch33_confs'], [batch_size, -1])
    # locs22 = tf.reshape(endpoints['branch22_locs'], [batch_size, -1])
    # confs22 = tf.reshape(endpoints['branch22_confs'], [batch_size, -1])
    # locs11 = tf.reshape(endpoints['branch11_locs'], [batch_size, -1])
    # confs11 = tf.reshape(endpoints['branch11_confs'], [batch_size, -1])

    locs = Concatenate(axis=1)([locs88, locs66, locs44, locs33, locs22, locs11])


    confs = Concatenate(axis=1)([confs88, confs66, confs44, confs33, confs22, confs11])
    # confs = Reshape((-1,))(confs)

    loc_confs = Concatenate(axis=2)([locs, confs])


    # locs = tf.concat([locs88, locs66, locs44, locs33, locs22, locs11], 1)
    # locs = tf.reshape(locs, [batch_size, -1, 4])
    #
    # confs = tf.concat([confs88, confs66, confs44, confs33, confs22, confs11], 1)
    # confs = tf.reshape(confs, [batch_size, -1, 1])
    # confs = tf.sigmoid(confs)
    #
    # loc_confs = tf.concat([locs, confs], 1)

    return loc_confs


def multibox_loss(y_true, y_pred):
    ground_boxes = y_true[:, :, :4]
    # ground_box = np.split(y_true, [5, y_len], axis=1)[0]
    locs = y_pred[:, :, :4]
    confs = y_pred[:, :, 4]
    # locs, confs = np.split(y_pred, [int(0.8*y_len), y_len], axis=1)
    # pred_boxes = np.split(locs, 1420, axis=1)

    # min_losses = K.placeholder(shape=(batchSize,))
    min_losses = []

    for b in range(batchSize):

        batch_gt = ground_boxes[b]
        batch_preds = locs[b] + pboxes
        batch_confs = K.clip(confs[b], 0.0001, 0.9999)

        conf_sum = K.sum(K.log(1 - batch_confs))
        conf_loss = -conf_sum + K.log(1-batch_confs) - K.log(batch_confs)

        loc_loss = 0.5 * K.sum(K.square(batch_gt - batch_preds), axis=1)

        min_loss = K.min(conf_loss + alpha * loc_loss)

        min_losses.append(min_loss)

        # batch_boxes = locs[b]
        # ground_box = ground_boxes[b]
        # batch_conf = confs[b]
        # conf_sum = K.sum(K.log(1 - batch_conf))
        #
        # min_loss = K.constant(100000.0)
        # # all_losses = K.placeholder(shape=(y_len,))
        # all_losses = np.zeros(y_len)
        #
        # min_loss = K.min(K.map_fn(batch_map, y_pred[b, :, :], axis=0))
        # for i in range(y_len):
        #
        #
        #     pred_box = batch_boxes[i, :]
        #     conf = batch_conf[i]
        #
        #     loss = (conf_loss(conf, conf_sum) +
        #             alpha * loc_loss(ground_box, pred_box))
        #
        #     print(loss.shape)
        #
        #     all_losses[i] = loss
        #     # if K.less(loss, min_loss):
        #     #     min_loss = loss
        #
        # min_losses[b] = K.min(K.variable(all_losses))

    min_losses_tensor = K.stack(min_losses)
    return min_losses_tensor


# create the base pre-trained model
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(width, height, 3))
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(height, width, 3))

# add a global spatial average pooling layer
x = base_model.output
loc_confs = bounding_box_prediction_layers(x, num_bboxes_per_cell, batchSize)


# this is the model we will train
model = Model(inputs=base_model.input, outputs=loc_confs)

for layer in base_model.layers:
    layer.trainable = False

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.00001), loss=multibox_loss, metrics=['acc'])

# assert(0 == 1)
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         zoom_range=0.2,
#         rotation_range=15,
#         horizontal_flip=True)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#         trainFolder,
#         target_size=(height, width),
#         batch_size=32)
#
# validation_generator = test_datagen.flow_from_directory(
#         validationFolder,
#         target_size=(height, width),
#         batch_size=32)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history2 = TrainingHistory()
history = model.fit_generator(imgGenerator(batchSize, True),
                              steps_per_epoch=int(float(len(indices_for_train)) / 32.0),
                              epochs=15,
                              validation_data=imgGenerator(batchSize, False),
                              validation_steps=int(float(len(indices_for_val)) / 32.0),
                              callbacks=[history2])

print(history.history)
print(history2.losses, history2.val_losses, history2.accs, history2.val_accs)

model.save('multibox1.h5')
