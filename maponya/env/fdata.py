import pandas as pd
import ta

DAY = 1440

def load_data():
    df = pd.read_csv('../fxt/env/datasets/fx_2018.csv', sep=';', 
        names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    df['MACD'] = ta.trend.macd(df['Close'], n_slow=52, n_fast=25, fillna=True)
    df['MA'] = ta.volatility.bollinger_mavg(df['Close'], n=20, fillna=True)
    df['EMA'] = ta.trend.ema_indicator(df['Close'], n=12, fillna=True)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], n=14, fillna=True)
    df['ROC'] = ta.momentum.roc(df['Close'], n=12, fillna=True)
    df['ICHI'] = ta.trend.ichimoku_a(df['High'], df['Low'], n1=9, n2=26, visual=False, fillna=True)

    return df[['Open', 'High', 'Low', 'Close', 'MACD', 'MA', 'EMA', 'ATR', 'ROC', 'ICHI']]