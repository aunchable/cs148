import os
import random
from PIL import Image
import numpy as np
import shutil

baseFolder = './CUB_200_2011/CUB_200_2011/images'
imageListFile = './CUB_200_2011/CUB_200_2011/images.txt'
splitListFile = './CUB_200_2011/CUB_200_2011/train_test_split.txt'

imageInfo = {}

paths_for_train = []
paths_for_val = []

f = open(imageListFile, 'r')
for line in f:
    [img_id, path] = line.split(' ')
    imageInfo[img_id] = path[:-1]

f = open(splitListFile, 'r')
for line in f:
    [img_id, img_class] = line.split(' ')
    if img_class[0] == '1':
        paths_for_train.append(imageInfo[img_id].split('/')[-1])
    else:
        paths_for_val.append(imageInfo[img_id].split('/')[-1])

shutil.copytree(baseFolder, './CUB_200_2011/CUB_200_2011/train')
shutil.copytree(baseFolder, './CUB_200_2011/CUB_200_2011/validation')

for path, subdirs, files in os.walk('./CUB_200_2011/CUB_200_2011/train'):
    for name in files:
        if name not in paths_for_train:
            os.remove(os.path.join(path, name))

for path, subdirs, files in os.walk('./CUB_200_2011/CUB_200_2011/validation'):
    for name in files:
        if name not in paths_for_val:
            os.remove(os.path.join(path, name))
