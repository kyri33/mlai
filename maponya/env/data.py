import pandas as pd
import ta

DAY = 1440

def load_data():
    df = pd.read_csv('../../fxt/env/datasets/fx_2018.csv', sep=';', 
        names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    df['MACD'] = ta.trend.macd(df['Close'], n_slow=52, n_fast=25, fillna=True)
    df['MA'] = ta.volatility.bolinger_mavg(df['Close'], n=20, fillna=True)
    df['EMA'] = ta.trend.ema_indicator(df['Close'], n=12, fillna=True)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], n=14, fillna=True)
    df['ROC'] = ta.momentum.roc(df['Close'], n=12, fillna=True)
    
    total_sets = len(df) // type
    sets = []
    for i in range(total_sets):
        sets.append(df.iloc[i * DAY : i * DAY + DAY])
    return sets