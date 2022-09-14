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
    RIGHT JOIN predictions_2017 as pd
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
        df.to_csv('zillow_project.csv')
    # change bedroom count to an integer
    df.bedroomcnt = df.bedroomcnt.astype(int)
    # change year built to an integer
    df.yearbuilt = df.yearbuilt.astype(int)
    # change fips to an integer
    df.fips = df.fips.astype(int)
    # rename columns for readability
    df = df.rename(columns={'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'calculatedfinishedsquarefeet': 'sqft', 
                        'taxvaluedollarcnt': 'tax_value', 'yearbuilt': 'year','lotsizesquarefeet':'lot_size', 
                        'regionidzip':'zipcode'})
    # remove rows with 6 or more bedrooms
    df = df[df['bedrooms'] < 6]
    # remove rows with 5 or more bathrooms
    df = df[df['bathrooms'] < 5]
    # remove rows with values less than or equal to 700 square feet
    df = df[df.sqft > 700]
    # remove rows with values greater than or equal to 11_000 square feet
    df = df[df.sqft < 11000]
    # remove rows with tax values greater than or equal to 700000
    df = df[df.tax_value < 700000]
    # remove rows with tax values less than or equal to 10000
    df = df[df.tax_value > 100000]
    # remove rows with a year less than or equal to 1899
    df = df[df.year > 1899]
    # remove rows with lot size less than 12000 square feet
    df = df[df.lot_size < 12000]
    # remove rows with lot size greater than 1000 square feet 
    df = df[df.lot_size > 1000]
    df.fips = df.fips.astype(str)
    # create dummies for the 'day' and 'time' columns
    dummy_df = pd.get_dummies(df[['fips']], dummy_na=False)
    # concatenate the dummy columns and the original dataframe
    df = pd.concat([df, dummy_df], axis=1)
    # create a new column for the total number of bedrooms and bathrooms
    df['bed_bath'] = df.bathrooms + df.bedrooms
    # convert fips to an integer
    df.fips = df.fips.astype(int)
    # create a new column for the difference between lot size and home size
    df['lot_minus_home'] = df.lot_size - df.sqft
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