import numpy as np
import pandas as pd
import scipy.stats as stats

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prnt_miss}).\
    reset_index().groupby(['num_cols_missing', 'percent_cols_missing']).count().\
    reset_index().rename(columns={'customer_id': 'count'})
    return rows_missing

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    prnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prnt_miss}).\
    reset_index().groupby(['num_rows_missing', 'percent_rows_missing']).count().reset_index().\
    rename(columns={'index': 'count'})
    return cols_missing

def summarize(df):
    print('DataFrame head: \n')
    print(df.head())
    print('----------')
    print('DataFrame info: \n')
    print(df.info())
    print('----------')
    print('DataFrame description: \n')
    print(df.describe())
    print('----------')
    print('Null value assessments: \n')
    print('Nulls by column: ', nulls_by_col(df))
    print('----------')
    print('Nulls by row: ', nulls_by_row(df))
    numerical_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in numerical_cols]
    print('----------')
    print('Value counts: \n')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
        print('-----')
    print('----------')
    print('Report Finished')


