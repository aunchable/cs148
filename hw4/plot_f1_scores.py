# Our numerical workhorses
import numpy as np

# Scikit-image submodules
import skimage.filters
import skimage.io
import skimage.morphology
from skimage import measure

import matplotlib.pyplot as plt

import os

import csv

counts = np.zeros(200)

validationFolder = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/validation/'
for subdir in os.listdir(validationFolder):
    if subdir[0] != '.':
        bird_class = int(subdir[:3]) - 1
        counts[bird_class] = len(os.listdir(os.path.join(validationFolder, subdir)))

iou_scores = []
with open('/Users/anshulramachandran/Desktop/ious.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for row in datareader:
        iou_scores.append([float(row[0]), float(row[1]), float(row[2])])

iou_thresh = 0.1
for iou_conf in iou_scores:
    if iou_conf[0] > iou_thresh:
        iou_conf[0] = 1
    else:
        iou_conf[0] = 0


iou_scores = np.asarray(iou_scores)
iou_scores = iou_scores[iou_scores[:,2].argsort()]

iou_birds = np.split(iou_scores, np.where(np.diff(iou_scores[:,2]))[0]+1)

peak_f1 = np.zeros(200)

for i in range(200):
    iou_bird = iou_birds[i]
    iou_bird = iou_bird[iou_bird[:,1].argsort()[::-1]]

    count_bird = counts[i]

    bird_f1 = []

    count_tp = 0
    j = 0
    while count_tp < count_bird and j < len(iou_bird):
        precision = float(np.sum([row[0] for row in iou_bird[:(j+1)]])) / float(j+1)
        recall = float(np.sum([row[0] for row in iou_bird[:(j+1)]])) / count_bird
        if precision == 0.0 and recall == 0.0:
            bird_f1.append(0.0)
        else:
            bird_f1.append(2 * precision * recall / (precision + recall))
        j += 1
        if iou_bird[j][0] == 1:
            count_tp += 1


    peak_f1[i] = max(bird_f1)

print(peak_f1)

xaxis = np.asarray(range(1,201))

plt.bar(xaxis, peak_f1, align='center', alpha=0.5)
plt.xlabel('Bird class label')
plt.ylabel('F1 score')
plt.title('Peak F1 score over the 200 bird species')

plt.savefig('./graphs/f1_1.png')
plt.clf()
plt.cla()
plt.close()

xy = np.vstack((xaxis, peak_f1)).T
xy = xy[xy[:,1].argsort()[::-1]]
xy = xy.T

plt.bar(xaxis, xy[1], align='center', alpha=0.5)
plt.ylabel('F1 score')
plt.title('Peak F1 score over the 200 bird species')

plt.savefig('./graphs/f1_2.png')
plt.clf()
plt.cla()
plt.close()
