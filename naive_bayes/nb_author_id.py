#!/usr/bin/python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

clf = GaussianNB()

features_train, features_test, labels_train, labels_test = preprocess()

clf.fit(features_train, labels_train)

predition = clf.predict(features_test)

print accuracy_score(predition, labels_test)