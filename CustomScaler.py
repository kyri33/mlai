import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CustomScaler():

    def fit(self, arr):
        scaled_arr = np.empty(arr.shape)
        colnum = 0
        for col in arr.T:
            print(col.reshape(1, -1))
            maxnum = np.amax(col)
            minnum = np.amin(col)
            maxmin = max(abs(maxnum), abs(minnum))
            print(maxmin)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(np.array([[maxmin], [-maxmin]]))
            col = scaler.transform(col.reshape(1, -1))
            print(col)

            scaled_arr[:, colnum] = col
            colnum += 1
            #scaler2 = MinMaxScaler()
            #scaler2.fit(col.reshape(1, -1))
            #print(scaler2.transform(col.reshape(1, -1)))
        print(scaled_arr)


arr = np.array([
    [1, 2, 12, 1],
    [2, 9, 4, 2],
    [-1, -2, -1, 3],
    [-4, -5, -2, 4],
    [-9, 3, -12, 5]
])

scaler = CustomScaler()
print(arr)
scaler.fit(arr)

minmax = MinMaxScaler(feature_range=(0, 1))
print("\n")
print(minmax.fit_transform(arr))