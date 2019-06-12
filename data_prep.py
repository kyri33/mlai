import numpy as np
from sklearn import preprocessing

input_data = np.array([[6.6, -2.9, 3.3],
                      [-1.2, 7.8, -6.1],
                      [3.9, 0.4, 2.1],
                      [7.3, -9.9, -4.5]])

def binarize_data(in_data):
    data_binarized = preprocessing.Binarizer(threshold=1).transform(in_data)
    print("\n Binarized Data:\n", data_binarized)

def mean_removal(in_data): # Centers the data around 0
    print(in_data)
    print("\nMean =", in_data.mean(axis=0))
    print("\nStd =", in_data.std(axis=0))
    data_scaled = preprocessing.scale(in_data)
    print("\nScaled Mean =", data_scaled.mean(axis=0))
    print("\nScaled std =", data_scaled.std(axis=0))
    print("\nData scaled ", data_scaled)

def scaling(in_data): # Don't understand this result
    scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,100))
    scaled_data = scaler_minmax.fit_transform(in_data)
    print(scaled_data)

def normalize(in_data): # Puts data on a common scale
    norm_l1 = preprocessing.normalize(in_data, norm='l1') # All values sum up to 1 (ignores outliers)
    norm_l2 = preprocessing.normalize(in_data, norm='l2') # Squares of values sum up to 1 (includes outliers)
    print("L1: ", norm_l1)
    print("l2: ", norm_l2)

#binarize_data(input_data)
#mean_removal(input_data)
#scaling(input_data)
normalize(input_data)
