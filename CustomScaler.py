import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CustomScaler():

    def fit(self, arr):
        for col in arr.T:
            print(col)


arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [-1, -2, -3],
    [-4, -5, -6]
])

scaler = CustomScaler()
print(arr)
scaler.fit(arr)