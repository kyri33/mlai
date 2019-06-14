import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y):
    # define min and max for X and y
    min_x, max_x = X[:,0].min() , X[:,0].max()
    min_y, max_y = X[:, 1].min(), X[:, 1].max()
    
    mesh_step_size = 1.0
    x_vals, y_vals= np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                np.arange(min_y, max_y, mesh_step_size))
    print(list(x_vals.ravel()))
    print(list(y_vals.ravel()))
    #print(np.c_[x_vals.ravel(), y_vals.ravel()])

X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5],
            [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])

visualize_classifier(0, X, 0)
