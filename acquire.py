import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
import acquire

####
#creating a connect function to connected to the code up servers
def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''initiates sql connection'''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#csv clean a downloaded version of the zillow data
def clean_zillow():
    '''Read zillow csv file into a pandas DataFrame,
    renamed all of the columuns, replace NaN values with 0 ,
    keep all the 0 values, convert all columns to int64,
    return cleaned zillow DataFrame'''
    df=pd.read_csv('zillow.csv')
    

    return df

#use to acquire from sql 
def sqlclean_zillow():
    query = """
    SELECT bedroomcnt,bathroomcnt,calculatedfinishedsquarefeet,taxvaluedollarcnt,yearbuilt
    ,fips,regionidcounty,regionidzip,numberofstories,latitude,longitude,logerror,taxamount FROM properties_2017
    JOIN propertylandusetype
    USING(propertylandusetypeid)
    JOIN predictions_2017
    USING(parcelid)
    WHERE propertylandusedesc like ('Single Family Residential') and  latitude IS NOT NULL
    AND longitude IS NOT NULL
    AND transactiondate <= '2017-12-31' """

    url = f"mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow"
    df = pd.read_sql(query,url)

    return df