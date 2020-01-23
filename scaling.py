import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

arr = np.array([[1], [2], [3], [4]])

scaler = StandardScaler()

print(scaler.fit_transform(arr))