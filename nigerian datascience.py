# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:42:28 2019

@author: OMOTESHO
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Datasets
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,[5,7,9,10,11]].values
y = dataset.iloc[:, 18].values

#splitting dataset to training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.45, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting logistic Regression to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns = ['Promoted_or_Not']).to_csv('Submission.csv')
#making Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)