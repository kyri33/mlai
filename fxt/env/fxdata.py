import pandas as pd

DAY = 1440
WEEK = DAY * 5
YEAR_RANGE = 2018

# TODO Different pairs

def load_data(group_by=DAY):

    # TODO Add concatenation for multiple years

    return load_year(2018, group_by)

def group(df, type=DAY):
    print(len(df))
    total_sets = len(df) // type
    sets = []
    for i in range(total_sets):
        sets.append(df.iloc[i * type : i * type + type])
    return sets

def load_year(year, group_by=DAY):
    df = pd.read_csv('./datasets/fx_' + str(year) + ".csv", sep=';', names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return group(df, group_by)
