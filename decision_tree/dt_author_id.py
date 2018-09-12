#!/usr/bin/python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

clf = DecisionTreeClassifier(random_state=0, min_samples_split=40)

features_train, features_test, labels_train, labels_test = preprocess()

clf.fit(features_train, labels_train)

prediction = clf.predict(features_test)

print accuracy_score(prediction, labels_test)

# percentile = 10
print len(features_train[0])  # 3785
print len(features_train)  # 15820

# percentile = 1
print len(features_train[0])  # 379
print len(features_train)  # 15820

