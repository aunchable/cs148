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

imgName = 'Black_Footed_Albatross_0039_796132.jpg'

print(name_id[imgName])
image = Image.open(name_id[imgName][1])
info = partInfo[name_id[imgName][0]]
print(info)
if info[0][0] < info[1][0]:
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    info[0][0] = image.size[0] - info[0][0]
    info[1][0] = image.size[0] - info[1][0]

rotation_angle = np.arctan((info[0][1]-info[1][1])/(info[0][0]-info[1][0]))*180.0/3.14159265
print(rotation_angle)

image = image.rotate(rotation_angle)

scipy.misc.imsave('./example.jpg', image)


# (currw, currh) = image.size
# if currw >= currh:
#     image = image.resize((width, int(float(currh) * float(width) / float(currw))))
# else:
#     image = image.resize((int(float(currw) * float(height) / float(currh)), height))
# # image.thumbnail((width, height), Image.ANTIALIAS)
# # return image
# background = Image.new('RGB', (width, height), (0, 0, 0))
# background.paste(
#     image, (int((width - image.size[0]) / 2), int((height - image.size[1]) / 2))
# )
#
# scipy.misc.imsave('./example.jpg', background)
