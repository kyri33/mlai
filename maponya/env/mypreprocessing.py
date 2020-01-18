import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CustomMinMaxScaler():

    def fit(self, arr):
        self.scalers = []
        for col in arr.T:
            maxnum = np.amax(col)
            minnum = np.amin(col)
            maxmin = max(abs(maxnum), abs(minnum))
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(np.array([[maxmin], [-maxmin]]))
            self.scalers.append(scaler)

    def transform(self, arr):
        colnum = 0
        scaled_arr = np.empty(arr.shape)
        for col in arr.T:
            col = self.scalers[colnum].transform(col.reshape(1, -1))
            scaled_arr[:, colnum] = col
            colnum += 1
        
        return scaled_arr

    def fit_transform(self, arr):
        self.fit(arr)
        return self.transform(arr)


'''
arr = np.array([
    [1,     2,      12,     1],
    [2,     9,      13,      2],
    [-1,    -2,     14,     3],
    [-4,    -5,     15,     4],
    [-9,    3,      -2,    5]
])

scaler = CustomMinMaxScaler()
scaler.fit(arr)
print(scaler.transform(arr))

arr = np.array([[11], [12], [13], [14], [15]])
minmax = MinMaxScaler()
minmax.fit(arr)
arr2 = [[-1], [-2], [13], [15], [17]]
print(minmax.transform(arr2))
'''