from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.random as r

digits = load_digits()

X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)

y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

# Using 10 output nodes for the digits 0-9 as a vector, therefore need to convert y_train to vector:

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect

y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)

# 64 input nodes, 10 ouput nodes and a hidden layer of 30
nn_structure = [64, 30, 10]

# activation function
def f(x):
    return 1 / (1 + np.exp(-x))

def f_deriv(x):
    return f(x) * (1 - f(x))

def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l - 1]))
        b[l] = r.random_sample((nn_structure[l]),)
    return W, b

w, b = setup_and_init_weights(nn_structure)
print(w[1][0])