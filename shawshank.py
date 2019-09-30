import pandas as pd
import matplotlib.pyplot as plt
import string

data = pd.read_csv("datasets/jse/quotes.csv", sep=";")

def clean_commas(arr):
    return [float(val.replace(',', '.')) for val in arr]

data['Close'] = clean_commas(data['Close'])
data['Open'] = clean_commas(data['Open'])
data['High'] = clean_commas(data['High'])
data['Low'] = clean_commas(data['Low'])

print(''.join(data['Volume'][0].split()))
data['Volume'] = [int(''.join(v.split())) for v in data['Volume']]

print(data.head())

plt.plot(data['Close'], 'r')
#plt.plot(data['Volume'], 'b')
plt.show()