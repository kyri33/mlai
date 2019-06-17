import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

text = np.loadtxt('data_perceptron.txt')

data = text[:, :2]
labels = text[:,2].reshape((text.shape[0], 1))
print(labels)

plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel("Dimension 1")
plt.ylabel("Dimenstion 2")
plt.title('Input Data')

dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1

num_output = labels.shape[1]
