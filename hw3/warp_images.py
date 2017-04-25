import os
import random
from PIL import Image
import numpy as np
import shutil
import scipy.misc

baseImgPath = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images'
partLocsFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/parts/part_locs.txt'
imageListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/images.txt'

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
    name_id[path[:-1].split('/')[-1]] = [img_id, os.path.join(baseImgPath, path[:-1])]

imgName = 'Black_Footed_Albatross_0078_796126.jpg'

# print(name_id[imgName])
image = Image.open(name_id[imgName][1])
info = partInfo[name_id[imgName][0]]
# print(info)
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

rotation_angle = np.arctan((info[0][1]-info[1][1])/(info[0][0]-info[1][0]))*180.0/3.14159265
# print(rotation_angle)

image2 = background.rotate(rotation_angle)

# print(distBetweenPoints)

bbox = (int(image2.size[0] / 2.0 - 2 * distBetweenPoints), int(image2.size[1] / 2.0 - 2 * distBetweenPoints),
        int(image2.size[0] / 2.0 + 2 * distBetweenPoints), int(image2.size[1] / 2.0 + 2 * distBetweenPoints))

image2 = image2.crop(bbox)
# print(image2)

scipy.misc.imsave('./example2.jpg', image2)
