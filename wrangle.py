import pandas as pd
import numpy as np
import env
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, RobustScaler

def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''This function uses credentials from an env file to log into a database'''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_db():
    '''The function uses the get_connection function to connect to a database and retrieve the zillow dataset'''
    
    zillow = pd.read_sql('''
    SELECT p.id, p.parcelid, pd.logerror, pd.transactiondate, p.airconditioningtypeid, ac.airconditioningdesc, 
    p.architecturalstyletypeid, a.architecturalstyledesc, p.basementsqft, p.bathroomcnt, p.bedroomcnt, 
    p.buildingclasstypeid, b.buildingclassdesc, p.buildingqualitytypeid, p.calculatedbathnbr, p.decktypeid, 
    p.finishedfloor1squarefeet, p.calculatedfinishedsquarefeet, p.finishedsquarefeet12, p.finishedsquarefeet13, 
    p.finishedsquarefeet15, p.finishedsquarefeet50, p.finishedsquarefeet6, p.fips, p.fireplacecnt, p.fullbathcnt, 
    p.garagecarcnt, p.garagetotalsqft, p.hashottuborspa, p.heatingorsystemtypeid, h.heatingorsystemdesc, p.latitude, 
    p.longitude, p.lotsizesquarefeet, p.poolcnt, p.poolsizesum, p.pooltypeid10, p.pooltypeid2, p.pooltypeid7, 
    p.propertycountylandusecode, p.propertylandusetypeid, p.propertyzoningdesc, p.rawcensustractandblock, 
    p.regionidcity, p.regionidneighborhood, p.regionidzip, p.roomcnt, p.storytypeid, p.threequarterbathnbr, 
    p.typeconstructiontypeid, p.unitcnt, p.yardbuildingsqft17, p.yardbuildingsqft26, p.yearbuilt, p.numberofstories, 
    p.fireplaceflag, p.structuretaxvaluedollarcnt, p.taxvaluedollarcnt, p.assessmentyear, p.landtaxvaluedollarcnt, 
    p.taxamount, p.taxdelinquencyflag, p.taxdelinquencyyear, p.censustractandblock

    FROM properties_2017 as p
    INNER JOIN predictions_2017 as pd
    ON p.id = pd.id
    LEFT JOIN airconditioningtype as ac
    ON p.airconditioningtypeid = ac.airconditioningtypeid
    LEFT JOIN architecturalstyletype as a
    ON p.architecturalstyletypeid = a.architecturalstyletypeid
    LEFT JOIN buildingclasstype as b
    ON p.buildingclasstypeid = b.buildingclasstypeid
    LEFT JOIN heatingorsystemtype as h
    ON p.heatingorsystemtypeid = h.heatingorsystemtypeid
    LEFT JOIN propertylandusetype as l
    ON p.propertylandusetypeid = l.propertylandusetypeid
    LEFT JOIN storytype as s
    ON p.storytypeid = s.storytypeid
    LEFT JOIN typeconstructiontype as t
    ON p.typeconstructiontypeid = t.typeconstructiontypeid
    LEFT JOIN unique_properties as u
    ON p.parcelid = u.parcelid
    WHERE p.latitude IS NOT NULL
    AND p.longitude IS NOT NULL
    AND p.propertylandusetypeid = 261

    ;''', get_connection('zillow'))
    return zillow

def get_zillow_data():
    ''' This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.'''
    if os.path.isfile('zillow.csv'):
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)     
    else:   
        # Read fresh data from db into a DataFrame
        df = new_zillow_db()
        # Cache data
        df.to_csv('zillow.csv')

def wrangle_zillow():
    '''This function acquires the zillow dataset from the Codeup database using a SQL query and returns a cleaned
    dataframe from a csv file. Observations with null values are dropped and column names are changed for
    readability. Values expected as integers are converted to integer types (year, bedrooms, fips).'''
    if os.path.isfile('zillow.csv'):
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)     
    else:   
        # Read fresh data from db into a DataFrame
        df = new_zillow_db()
        # Cache data
        df.to_csv('zillow.csv')
    
    return df

def split_data(df):
    '''This function takes in a dataframe and returns three dataframes, a training dataframe with 60 percent of the data, 
        a validate dataframe with 20 percent of the data and test dataframe with 20 percent of the data.'''
    # split the dataset into two, with 80 percent of the observations in train and 20 percent in test
    train, test = train_test_split(df, test_size=.2, random_state=217)
    # split the train again into two sets, using a 75/25 percent split
    train, validate = train_test_split(train, test_size=.25, random_state=217)
    # return three datasets, train (60 percent of total), validate (20 percent of total), and test (20 percent of total)
    return train, validate, test

def quantile_scaler_norm(a,b,c):
    '''This function applies the .QuantileTransformer method from sklearn to three arguments, a, b, and c,
    (X_train, X_validate, and X_test) and returns the scaled versions of each variable.'''
    # make the scaler
    scaler = QuantileTransformer(output_distribution='normal')
    # fit and transform the X_train variable
    X_train_quantile = pd.DataFrame(scaler.fit_transform(a))
    # transform the X_validate variable
    X_validate_quantile = pd.DataFrame(scaler.transform(b))
    # transform the X_test variable
    X_test_quantile = pd.DataFrame(scaler.transform(c))
    # return three variables, one for each newly scaled variable
    return X_train_quantile, X_validate_quantile, X_test_quantile

def quantile_scaler(a,b,c):
    '''This function applies the .QuantileTransformer method from sklearn to three arguments, a, b, and c,
    (X_train, X_validate, and X_test) and returns the scaled versions of each variable.'''
    # make the scaler
    scaler = QuantileTransformer()
    # fit and transform the X_train variable
    X_train_quantile = pd.DataFrame(scaler.fit_transform(a))
    # transform the X_validate variable
    X_validate_quantile = pd.DataFrame(scaler.transform(b))
    # transform the X_test variable
    X_test_quantile = pd.DataFrame(scaler.transform(c))
    # return three variables, one for each newly scaled variable
    return X_train_quantile, X_validate_quantile, X_test_quantile

def standard_scaler(a,b,c):
    '''This function applies the .StandardScaler method from sklearn to three arguments, a, b, and c, 
    (X_train, X_validate, and X_test) and returns the scaled versions of each variable.'''
    # make the scaler
    scaler = StandardScaler()
    # fit and transform the X_train data
    X_train_standard = pd.DataFrame(scaler.fit_transform(a))
    # transform the X_validate data
    X_validate_standard = pd.DataFrame(scaler.transform(b))
    # transform the X_test data
    X_test_standard = pd.DataFrame(scaler.transform(c))
    # return the scaled data for each renamed variable
    return X_train_standard, X_validate_standard, X_test_standard

def minmax_scaler(a,b,c):
    '''This function applies the .MinMaxScaler method from sklearn to three arguments, a, b, and c,
    (X_train, X_validate, and X_test) and returns the scaled versions of each variable.'''
    # make the scaler
    scaler = MinMaxScaler()
    # fit and transform the X_train data
    X_train_scaled = pd.DataFrame(scaler.fit_transform(a))
    # transform the X_validate data
    X_validate_scaled = pd.DataFrame(scaler.transform(b))
    # transform the X_test data
    X_test_scaled = pd.DataFrame(scaler.transform(c))
    # return the scaled data for each renamed variable
    return X_train_scaled, X_validate_scaled, X_test_scaled

def robust_scaler(a,b,c):
    '''This function applies the .RobustScaler method from sklearn to three arguments, a, b, and c,
    (X_train, X_validate, and X_test) and returns the scaled versions of each variable.'''
    # make the scaler
    scaler = RobustScaler()
    # fit and transform the X_train data
    X_train_robust = pd.DataFrame(scaler.fit_transform(a))
    # transform the X_validate data
    X_validate_robust = pd.DataFrame(scaler.transform(b))
    # transform the X_test data
    X_test_robust = pd.DataFrame(scaler.transform(c))
    # return the scaled data for each renamed variable
    return X_train_robust, X_validate_robust, X_test_robust

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
    numerical_cols = [col for col in df.columns if df[col].dtypes != 'O']
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

def get_upper_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k*iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    return df

def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, 
                          prop_required_columns=0.5, 
                          prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df

def split_data_strat(df, column):
    '''This function takes in two arguments, a dataframe and a string. The string argument is the name of the
        column that will be used to stratify the train_test_split. The function returns three dataframes, a 
        training dataframe with 60 percent of the data, a validate dataframe with 20 percent of the data and test
        dataframe with 20 percent of the data.'''
    train, test = train_test_split(df, test_size=.2, random_state=217, stratify=df[column])
    train, validate = train_test_split(train, test_size=.25, random_state=217, stratify=train[column])
    return train, validate, test

