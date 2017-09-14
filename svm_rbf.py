# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:46:34 2017

@author: Arvind
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:26:42 2017

@author: Arvind
"""

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData(5000)


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=1000000)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)


#### store your predictions in a list named pred





from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

pic = prettyPicture(clf, features_test, labels_test, "svm_rbf")

def submitAccuracy():
    return acc