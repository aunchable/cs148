# Our numerical workhorses
import numpy as np

# Scikit-image submodules
import skimage.filters
import skimage.io
import skimage.morphology
from skimage import measure

import matplotlib.pyplot as plt

import csv

iou_scores = []
with open('/Users/anshulramachandran/Desktop/ious.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for row in datareader:
        iou_scores.append([float(row[0]), float(row[1])])

iou_thresh = 0.2
for iou_conf in iou_scores:
    if iou_conf[0] > iou_thresh:
        iou_conf[0] = 1
    else:
        iou_conf[0] = 0

total_correct_positives = 5794.0

precision = []
recall = []

count_tp = 0
i = 0
while count_tp < total_correct_positives and i < len(iou_scores):
    precision.append(float(np.sum([row[0] for row in iou_scores[:(i+1)]])) / float(i+1))
    recall.append(float(np.sum([row[0] for row in iou_scores[:(i+1)]])) / total_correct_positives)
    i += 1
    if iou_scores[i][0] == 1:
        count_tp += 1

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('Precision-recall curve for IOU threhold = 0.2')

plt.savefig('./graphs/pr2.png')
plt.clf()
plt.cla()
plt.close()
