import pandas as pd
import matplotlib.pyplot as plt
import string
import numpy as np
from tqdm import tqdm

data = pd.read_csv("datasets/jse/quotes.csv", sep=";")

def clean_commas(arr):
    return [float(val.replace(',', '.')) for val in arr]

data['Close'] = clean_commas(data['Close'])
data['Open'] = clean_commas(data['Open'])
data['High'] = clean_commas(data['High'])
data['Low'] = clean_commas(data['Low'])

print(''.join(data['Volume'][0].split()))
data['Volume'] = [int(''.join(v.split())) for v in data['Volume']]

print(data.tail())

#plt.plot(data['Close'], 'r')
#plt.plot(data['Volume'], 'b')
#plt.show()


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

train_cols = ["Open", "High", "Low", "Close", "Volume"]
d_train, d_test = train_test_split(data, train_size = 0.8, test_size = 0.2, shuffle = False)

x = d_train.loc[:, train_cols].values
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x)
print(x_train.shape)
x_test = scaler.transform(d_test.loc[:, train_cols].values)

TIME_STEPS = 10

def build_timeseries(mat, y_col_index): # Puts values into 3D array
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in tqdm(range(dim_0)):
        x[i] = mat[i : TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    print("length of time-series i/o", x.shape, y.shape)
    return x,y

def trim_dataset(mat, batch_size):
    no_rows_drop = mat.shape[0] % batch_size

    if (no_rows_drop > 0):
        return mat[:-no_rows_drop]
    else:
        return mat

x_t, y_t = build_timeseries(x_train, 3)

