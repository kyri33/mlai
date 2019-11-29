import pandas as pd

def load_year(year):
    df = pd.read_csv('./datasets/fx_' + str(year) + ".csv", sep=';', names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    print(df)