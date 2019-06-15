import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_validate

input_file = 'adult.data'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '>=50K' and count_class1 < max_datapoints:
            count_class1 += 1
            X.append(data)
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            count_class2 += 1
            X.append(data)
    print(X)
    X = np.array(X)
    label_encoder = []
    X_encoded = np.empty(X.shape)
    print(X.shape)
    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:,i] = X[:,i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            X_encoded[:,i] = label_encoder[-1].fit_transform(X[:,i])

    print(X_encoded)
