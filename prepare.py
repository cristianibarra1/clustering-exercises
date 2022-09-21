import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
from env import user, password, host
import acquire
import prepare
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os 

##preparing data
def prep_zillow(df):
    '''prep the zzillow dataset by renaming the columns and 
    creating two new columns name decade and totalrooms
    i used df.drop to drop all of the null in this dataset 
    converted fips and year as objects'''
    #renaming all the columns again
    df = df.rename(columns={'bedroomcnt': 'Bedrooms', 'bathroomcnt': 'Bathrooms','calculatedfinishedsquarefeet':'Squarefeet',    "taxvaluedollarcnt":'TaxesTotal','yearbuilt':'Year','fips':'Fips','regionidcounty':'County','regionidzip':'Zip','numberofstories':'Stories','parcelid':'Parcelid','logerror':'Log_error','taxamount':'Taxamount'})
    #changing these column into objects
    df.Fips = df.Fips.astype(object)
    df.Year = df.Year.astype(object)
    df.Stories = df.Stories.astype(object)
    #creating a column name total rooms by accounting bath and bed rooms together
    df['TotalRooms'] = df['Bathrooms']+df['Bedrooms']
    #was trying to drop the null or replace them
    df=df.replace('NaN','0')
    df = df.drop(columns=('Stories'))
    things = ['Year', 'TaxesTotal', 'Squarefeet']
    for col in things:
        q1,q3 = df[col].quantile([.25,.75])
        iqr = q3-q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr

        df = df[(df[col] > lower) & (df[col] < upper)]
    df['Decade'] = pd.cut(df.Year, bins=[1800,1850,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020],labels=['1800', '1850', '1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970','1980','1990','2000','2010'])
    df.Decade = df.Decade.astype(float)
    #created a column connecting years into decades plus drop nulls
    df=df.replace('','0')
    df = df.fillna(0)
    #making decade into a int
    df.Decade = df.Decade.astype(int)
    df['county'] = df.Fips.apply(lambda x: 'orange' if x == 6059.0 else 'los_angeles' if x == 6037.0 else 'ventura')
    #re arrange the columns back into place
    df.columns=['Bedrooms','Bathrooms','Squarefeet','TaxesTotal','Year','Fips','County','Zip','latitude','longitude','Log_error','Taxamount','TotalRooms','Decade','location']
    return df
     
def my_train_test_split(df):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123)
    train, validate = train_test_split(train, test_size = .25, random_state=123)
    
    return train, validate, test



def scale_data(train, val, test, cols_to_scale):
    '''scaled the data by creating train_scaled val_scaled test_scaled
    copying from orginal train,test,val
    fitting 
    creating a dataframe from the data '''
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[cols_to_scale])
    
    train_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(train[cols_to_scale]),
                                               columns = train[cols_to_scale].columns.values).set_index([train.index.values])
    val_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(val[cols_to_scale]),
                                               columns = val[cols_to_scale].columns.values).set_index([val.index.values])
    test_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(test[cols_to_scale]),
                                               columns = test[cols_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, val_scaled, test_scaled





def model_setup(train_scaled, train, val_scaled, val, test_scaled, test):
    '''mdoel from train test val'''

    # Set up X and y values for modeling
    X_train, y_train = train_scaled.drop(columns=['Log_error','location','Decade']), train.Log_error
    X_val, y_val = val_scaled.drop(columns=['Log_error','location','Decade']), val.Log_error
    X_test, y_test = test_scaled.drop(columns=['Log_error','location','Decade']), test.Log_error

    # make them a dataframes
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test