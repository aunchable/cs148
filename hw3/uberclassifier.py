import numpy as np
from sklearn import svm
import os
import random
import csv
import scipy.misc

trainX = []
trainY = []
testX = []
testY = []


with open('/Users/anshulramachandran/Desktop/all_train.csv', newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for row in filereader:
        trainX.append([float(val) for val in row[1:]])
        trainY.append(int(row[0]))

with open('/Users/anshulramachandran/Desktop/all_validation.csv', newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',')
    for row in filereader:
        testX.append([float(val) for val in row[1:]])
        testY.append(int(row[0]))

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(trainX, trainY)

train_acc = clf.score(trainX, trainY)
test_acc = clf.score(testX, testY)
print(train_acc, test_acc)

validation_predictions = clf.predict(testX)

# Generate confusion matrix
confusion_matrix = np.zeros(shape=(200,200))

for i in range(len(testY)):
    class_true = testY[i]
    class_pred = validation_predictions[i]
    confusion_matrix[class_true][class_pred] += 1

confusion_matrix = confusion_matrix.astype(int)
confusion_matrix = -confusion_matrix
confusion_matrix_img = np.zeros(shape=(1000, 1000))
for i in range(len(confusion_matrix_img)):
    for j in range(len(confusion_matrix_img[0])):
        confusion_matrix_img[i][j] = confusion_matrix[int(i/5.0), int(j/5.0)]
scipy.misc.imsave('./graphs/confusion_matrix_uber.jpg', confusion_matrix_img)
for i in range(200):
    confusion_matrix[i][i] = 0
confusion_matrix_img = np.zeros(shape=(1000, 1000))
for i in range(len(confusion_matrix_img)):
    for j in range(len(confusion_matrix_img[0])):
        confusion_matrix_img[i][j] = confusion_matrix[int(i/5.0), int(j/5.0)]
scipy.misc.imsave('./graphs/confusion_matrix_uber_no_diagonal.jpg', confusion_matrix_img)
