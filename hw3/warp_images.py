import os
import random
from PIL import Image
import numpy as np
import shutil
import scipy.misc

trainFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/train3'
validationFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation3'
partLocsFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/parts/part_locs.txt'
imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'

# trainFolder = './CUB_200_2011/CUB_200_2011/train3'
# validationFolder = './CUB_200_2011/CUB_200_2011/validation3'
# partLocsFile = './CUB_200_2011/CUB_200_2011/parts/part_locs.txt'
# imageListFile = './CUB_200_2011/CUB_200_2011/images.txt'

imageInfo = {}
partInfo = {}
name_id = {}

f = open(partLocsFile, 'r')
for line in f:
    [iid, partid, x1, y1, present] = line.split(' ')
    imageInfo[iid + '_' + partid] = [float(x1), float(y1)]

for i in range(1, 11789):
    if imageInfo[str(i) + '_11'][0] != 0:
        partInfo[str(i)] = [imageInfo[str(i) + '_2'], imageInfo[str(i) + '_11'], 0]
    else:
        partInfo[str(i)] = [imageInfo[str(i) + '_2'], imageInfo[str(i) + '_7'], 1]


f = open(imageListFile, 'r')
for line in f:
    [img_id, path] = line.split(' ')
    name_id[path[:-1].split('/')[-1]] = img_id


def warp_image(imgName, imgPath):

    image = Image.open(imgPath)
    info = partInfo[name_id[imgName]]

    if info[0][0] == 0 or info[0][1] == 0 or info[1][0] == 0 or info[1][1] == 0:
        return image

    if info[0][0] < info[1][0]:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        info[0][0] = image.size[0] - info[0][0]
        info[1][0] = image.size[0] - info[1][0]

    distBetweenPoints = np.sqrt(pow(info[1][0] - info[0][0], 2) + pow(info[1][1] - info[0][1], 2))

    distToTop = info[1][1]
    distToBot = image.size[1] - info[1][1]
    distToLeft = info[1][0]
    distToRight = image.size[0] - info[1][0]


    padTop = max(0, int(distToBot - distToTop))
    padBot = max(0, int(distToTop - distToBot))
    padLeft = max(0, int(distToRight - distToLeft))
    padRight = max(0, int(distToLeft - distToRight))

    background = Image.new('RGB', (padLeft + image.size[0] + padRight, padTop + image.size[1] + padBot), (0, 0, 0))
    background.paste(
        image, (padLeft, padTop)
    )

    try:
        rotation_angle = np.arctan((info[0][1]-info[1][1])/(info[0][0]-info[1][0]))*180.0/3.14159265
        image2 = background.rotate(rotation_angle)
    except ZeroDivisionError:
        image2 = background

    bbox = (int(image2.size[0] / 2.0 - 2 * distBetweenPoints), int(image2.size[1] / 2.0 - 2 * distBetweenPoints),
            int(image2.size[0] / 2.0 + 2 * distBetweenPoints), int(image2.size[1] / 2.0 + 2 * distBetweenPoints))

    image2 = image2.crop(bbox)

    return image2


for path, subdirs, files in os.walk(trainFolder):
    for name in files:
        if name[0] != '.':
            new_img = warp_image(name, os.path.join(path, name))
            scipy.misc.imsave(os.path.join(path, name), new_img)

for path, subdirs, files in os.walk(validationFolder):
    for name in files:
        if name[0] != '.':
            new_img = warp_image(name, os.path.join(path, name))
            scipy.misc.imsave(os.path.join(path, name), new_img)
