#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

m = int(len(features_train))
n = int(len(labels_train))

features_train = features_train[:m]
labels_train = labels_train[:n]


#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
t0 = time()
clf = SVC(kernel='rbf', C= 10000.0)
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
predict = clf.predict(features_test)
print ("prediction time:", round(time()-t0, 3), "s")
# print(clf.score(features_test, labels_test))
# print(accuracy_score(predict, labels_test))
number =0
print(len(features_test))
num = 1
for i in range(len(predict)):
    if predict[i] == 1 :
        number+=1
print(number)
#########################################################


