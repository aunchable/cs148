import numpy as np
from sklearn import svm
import os
import random
import csv

labelListFile = '/Users/anshulramachandran/Documents/Year3 Q3/CS148/CUB_200_2011/CUB_200_2011/image_class_labels.txt'

imageInfo = {}
trainEmbeddings = {}
validationEmbeddings = {}

f = open(labelListFile, 'r')
for line in f:
    [img_id, label] = line.split(' ')
    imageInfo[img_id] = label[:-1]

with open('/Users/anshulramachandran/Desktop/train1.csv', newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for row in filereader:
        trainEmbeddings[row[0]] = [float(val) for val in row[1:]]

with open('/Users/anshulramachandran/Desktop/validation1.csv', newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for row in filereader:
        validationEmbeddings[row[0]] = [float(val) for val in row[1:]]

with open('/Users/anshulramachandran/Desktop/train2.csv', newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for row in filereader:
        trainEmbeddings[row[0]] = trainEmbeddings[row[0]].append([float(val) for val in row[1:]])

with open('/Users/anshulramachandran/Desktop/validation2.csv', newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for row in filereader:
        validationEmbeddings[row[0]] = validationEmbeddings[row[0]].append([float(val) for val in row[1:]])

with open('/Users/anshulramachandran/Desktop/train3.csv', newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for row in filereader:
        trainEmbeddings[row[0]] = trainEmbeddings[row[0]].append([float(val) for val in row[1:]])

with open('/Users/anshulramachandran/Desktop/validation3.csv', newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for row in filereader:
        validationEmbeddings[row[0]] = validationEmbeddings[row[0]].append([float(val) for val in row[1:]])

train_rows = []
for k, v in trainEmbeddings.items():
    train_rows.append(v.insert(0, int(imageInfo[k])))

validation_rows = []
for k, v in validationEmbeddings.items():
    validation_rows.append(v.insert(0, int(imageInfo[k])))

with open('/Users/anshulramachandran/Desktop/all_train.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerows(train_rows)

with open('/Users/anshulramachandran/Desktop/all_validation.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerows(validation_rows)
