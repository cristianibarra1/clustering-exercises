import pandas as pd
import numpy as np 
import acquire
import prepare
import cluster
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
# Viz imports
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Modeling imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
# for modeling and evaluation
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures     
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
warnings.filterwarnings("ignore")
import scipy.stats as stats
import statistics
from sklearn.cluster import KMeans
# Custom module imports
import acquire
import prepare
import modeling
α = .05
alpha= .05

#acquire-----------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------   
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




#prepare-----------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------   
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






#modeling-----------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------   
    
def Modeling_function(X_train, y_train, X_val, y_val):
    ''' 
    This function takes in the X and y objects and then runs the following models:
    Using y_train mean to acquire baseline,
    LarsLasso Alpha=1,
    Quadratic Linear Regression
    
    Returns a DataFrame with the results.
    '''
    #most models are at 2 or 1
    # Baseline Model
    pred_mean = y_train.Log_error.mean()
    y_train['pred_mean'] = pred_mean
    y_val['pred_mean'] = pred_mean
    rmse_train = mean_squared_error(y_train.Log_error, y_train.pred_mean, squared=False)
    rmse_val = mean_squared_error(y_val.Log_error, y_val.pred_mean, squared=False)

    # save the results
    metrics = pd.DataFrame(data=[{
        'Model': 'baseline_mean',
        'Train_rmse': rmse_train,
        'Train_r2': explained_variance_score(y_train.Log_error, y_train.pred_mean),
        'Val_rmse': rmse_val,
        'Val_r2': explained_variance_score(y_val.Log_error, y_val.pred_mean)}])

    # LassoLars Model
    # run the model
    lars = LassoLars(alpha=2)
    lars.fit(X_train, y_train.Log_error)
    y_train['pred_lars'] = lars.predict(X_train)
    rmse_train = mean_squared_error(y_train.Log_error, y_train.pred_lars, squared=False)
    y_val['pred_lars'] = lars.predict(X_val)
    rmse_val = mean_squared_error(y_val.Log_error, y_val.pred_lars, squared=False)

    # save the results
    metrics = metrics.append({
        'Model': 'Lars_alpha(2)',
        'Train_rmse': rmse_train,
        'Train_r2': explained_variance_score(y_train.Log_error, y_train.pred_lars),
        'Val_rmse': rmse_val,
        'Val_r2': explained_variance_score(y_val.Log_error, y_val.pred_lars)}, ignore_index=True)

    # Polynomial Models
    # set up the model
    pf = PolynomialFeatures(degree=1)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)
    
    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.Log_error)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.Log_error, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.Log_error, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'Model': 'Depth(1)',
        'Train_rmse': rmse_train,
        'Train_r2': explained_variance_score(y_train.Log_error, y_train.pred_lm2),
        'Val_rmse': rmse_val,
        'Val_r2': explained_variance_score(y_val.Log_error, y_val.pred_lm2)}, ignore_index=True)

    # set up the model
    pf = PolynomialFeatures(degree=2)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.Log_error)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.Log_error, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.Log_error, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'Model': 'Depth(2)',
        'Train_rmse': rmse_train,
        'Train_r2': explained_variance_score(y_train.Log_error, y_train.pred_lm2),
        'Val_rmse': rmse_val,
        'Val_r2': explained_variance_score(y_val.Log_error, y_val.pred_lm2)}, ignore_index=True)
    # set up the model
    pf = PolynomialFeatures(degree=3)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.Log_error)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.Log_error, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.Log_error, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'Model': 'Depth(3)',
        'Train_rmse': rmse_train,
        'Train_r2': explained_variance_score(y_train.Log_error, y_train.pred_lm2),
        'Val_rmse': rmse_val,
        'Val_r2': explained_variance_score(y_val.Log_error, y_val.pred_lm2)}, ignore_index=True)
    # set up the model
    pf = PolynomialFeatures(degree=4)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.Log_error)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.Log_error, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.Log_error, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'Model': 'Depth(4)',
        'Train_rmse': rmse_train,
        'Train_r2': explained_variance_score(y_train.Log_error, y_train.pred_lm2),
        'Val_rmse': rmse_val,
        'Val_r2': explained_variance_score(y_val.Log_error, y_val.pred_lm2)}, ignore_index=True)

    return metrics


def modeling_best(X_train, y_train, X_val, y_val, X_test, y_test):
    ''''modeling the best model aka polynomial feature degree 2'''
    # set up the model
    pf = PolynomialFeatures(degree=2)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)
    X_test_d2 = pf.transform(X_test)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.Log_error)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.Log_error, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.Log_error, y_val.pred_lm2, squared=False)
    y_test['pred_lm2'] = lm2.predict(X_test_d2)
    rmse_test = mean_squared_error(y_test.Log_error, y_test.pred_lm2, squared=False)
    # save the results
    results = pd.DataFrame({'test':{'Test_rmse': rmse_test,'Test_r2': explained_variance_score(y_test.Log_error,y_test.pred_lm2)}})
    results.dropna()
    
    return results

#clusters modeling------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_clusters(df, k, cluster_vars):
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 123)

    # fit to train
    kmeans.fit(df[cluster_vars])

    return kmeans


def cluster_plot_bedrooms(train_scaled):
    '''
    creates catplot for clusters and performs ANOVA test to determine significance
    accepts a dataframe and cluster variables, plots log error by cluster, and prints test result
    '''
    
    sns.catplot(x ="cluster",
             y ="Log_error",
             data = train_scaled, size=5)
    plt.title("Log Error by cluster")
    plt.show()

    cluster1 = train_scaled[train_scaled.cluster == 0]
    cluster2 = train_scaled[train_scaled.cluster == 1]
    cluster3 = train_scaled[train_scaled.cluster == 2]
    cluster4 = train_scaled[train_scaled.cluster == 3]
    cluster5 = train_scaled[train_scaled.cluster == 4]
    print('Clusters for Bedroom and Decade')
    print('H0:  The log error is not significantly different')
    print('Ha:  The logerrors is significantly different')

    alpha = .05
    f, p = stats.f_oneway(cluster1.Log_error, cluster2.Log_error, cluster3.Log_error, cluster4.Log_error, cluster5.Log_error) 
    if p < alpha:
        print("We reject the Null Hypothesis")
    else:
        print("We confirm the Null Hypothesis")
        
        
def elbow_bed(train_scaled):
    # create features for cluster
    cluster_vars = ['Bedrooms','Decade']

    return modeling.find_k(train_scaled, cluster_vars)

        
def bedcluster_graph(train_scaled):
    # set k and name cluster
    k=5
    cluster_name = 'bed_decade'
    cluster_vars = ['Bedrooms','Decade']
    # create clusters
    kmeans1 = modeling.create_clusters(train_scaled, k, cluster_vars)

    X1_train = train_scaled.copy()
    X1_train['cluster'] = kmeans1.predict(train_scaled[cluster_vars])

    return modeling.cluster_plot_bedrooms(X1_train)



#tax cluster------------------------------------------------------------------------------------------------------------------
        
    
    
def cluster_plot_tax(train_scaled):
    '''
    creates catplot for clusters and performs ANOVA test to determine significance
    accepts a dataframe and cluster variables, plots log error by cluster, and prints test result
    '''
    
    sns.catplot(x ="cluster",
             y ="Log_error",
             data = train_scaled, size=5)
    plt.title("Log Error by cluster")
    plt.show()

    cluster1 = train_scaled[train_scaled.cluster == 0]
    cluster2 = train_scaled[train_scaled.cluster == 1]
    cluster3 = train_scaled[train_scaled.cluster == 2]
    cluster4 = train_scaled[train_scaled.cluster == 3]
    cluster5 = train_scaled[train_scaled.cluster == 4]
    print('Clusters for TaxesTotal and Squarefeet')
    print('H0:  The log error is not significantly different')
    print('Ha:  The logerrors is significantly different')
    #stats test 
    alpha = .05
    f, p = stats.f_oneway(cluster1.Log_error, cluster2.Log_error, cluster3.Log_error, cluster4.Log_error, cluster5.Log_error) 
    if p < alpha:
        print("We reject the Null Hypothesis")
    else:
        print("We confirm the Null Hypothesis")
        
def taxcluster_elbow(train_scaled):
    '''elbow for tax cluster'''
    # create features for cluster
    cluster_vars = ['TaxesTotal','Squarefeet']

    return modeling.find_k(train_scaled, cluster_vars)

def taxcluster_graph(train_scaled):
    # set k and name cluster
    k=5
    cluster_name = 'bed_decade'
    cluster_vars = ['TaxesTotal','Squarefeet']
    # create clusters
    kmeans1 = modeling.create_clusters(train_scaled, k, cluster_vars)

    X1_train = train_scaled.copy()
    X1_train['cluster'] = kmeans1.predict(train_scaled[cluster_vars])

    return modeling.cluster_plot_tax(X1_train)

def find_k(train_scaled, cluster_vars):
    ''''function for creating a elbow'''
    sse = []
    k_range = range(2,20)
    for k in k_range:
        kmeans = KMeans(n_clusters=k)

        kmeans.fit(train_scaled[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
                             sse=sse[0:-1]
                             ))

    # plot k with inertia
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 15})
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title(f'The Elbow Method to find the optimal k\n for {cluster_vars}\nFor which k values do we see large decreases in SSE?')
    # plt.annotate('Elbow', xy=(5, k_comparisons_df.sse[4]),  
    #         xytext=(5, 5), textcoords='axes fraction',
    #         arrowprops=dict(facecolor='red', shrink=0.05),
    #         horizontalalignment='right', verticalalignment='top',
            # )
    plt.show()

    return 




#long and lat cluster---------------------------------------------------------------------------------------------------------




def create_clusters(df, k, cluster_vars):
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 123)

    # fit to train
    kmeans.fit(df[cluster_vars])

    return kmeans


def cluster_plot_lat(train_scaled):
    '''
    creates catplot for clusters and performs ANOVA test to determine significance
    accepts a dataframe and cluster variables, plots log error by cluster, and prints test result
    '''
    
    sns.catplot(x ="cluster",
             y ="Log_error",
             data = train_scaled, size=5)
    plt.title("Log Error by cluster")
    plt.show()

    cluster1 = train_scaled[train_scaled.cluster == 0]
    cluster2 = train_scaled[train_scaled.cluster == 1]
    cluster3 = train_scaled[train_scaled.cluster == 2]
    cluster4 = train_scaled[train_scaled.cluster == 3]
    cluster5 = train_scaled[train_scaled.cluster == 4]
    print('Clusters for lat and long')
    print('H0:  The log error is not significantly different')
    print('Ha:  The logerrors is significantly different')

    alpha = .05
    f, p = stats.f_oneway(cluster1.Log_error, cluster2.Log_error, cluster3.Log_error, cluster4.Log_error, cluster5.Log_error) 
    if p < alpha:
        print("We reject the Null Hypothesis")
    else:
        print("We confirm the Null Hypothesis")
        
        
def elbow_lat(train_scaled):
    # create features for cluster
    cluster_vars = ['latitude','longitude']

    return modeling.find_k(train_scaled, cluster_vars)

        
def latcluster_lat(train_scaled):
    # set k and name cluster
    k=5
    cluster_name = 'lat_long'
    cluster_vars = ['latitude','longitude']
    # create clusters
    kmeans1 = modeling.create_clusters(train_scaled, k, cluster_vars)

    X1_train = train_scaled.copy()
    X1_train['cluster'] = kmeans1.predict(train_scaled[cluster_vars])

    return modeling.cluster_plot_lat(X1_train)

def longandlat(train_scaled,val_scaled,train,val):
    cluster_vars = train_scaled[['Squarefeet', 'Year', 'TaxesTotal']]
    cluster_col_name = 'size_year_value'
    centroid_col_names = ['centroid_' + i for i in cluster_vars]
    optimal_k = cluster.elbow_method(cluster_vars)
    # Function to obtain:
    # The train clusters with their observations,
    # test clusters and their observations
    # and a df of the number of observations per cluster on train
    kmeans, train_clusters, val_cluster, cluster_counts = cluster.get_clusters_and_counts(5, ['Year','Squarefeet', 'TaxesTotal'],'size_year_value', train_scaled,val_scaled)
    # Function to obtain:
    # dataframe of the train clusters with their observations, 
    # test clusters and their observations
    # and a df of the number of observations per cluster on train. 
    X_train_scaled, train_scaled, X_test_scaled, test_scaled, centroids = cluster.append_clusters_and_centroids(train, train_scaled, train_clusters,val, val_scaled, val_cluster,cluster_col_name, centroid_col_names, kmeans)
    cluster.test_sig(train_scaled.size_year_value, train_scaled)
    plt.scatter(train_scaled.latitude, train_scaled.longitude, c=train_scaled.Bedrooms)
    plt.xlabel('lat and long')
    plt.ylabel('Log error')
    plt.title('Clusters using lat/long compared to Log error')
    plt.show


#graphs-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def county_log(train):
    '''location compared to log error '''
    #orange has the highest logerror out of all of them aka countys
    plt.figure(figsize=(17,10))
    sns.barplot(data=train,x='location', y='Log_error',hue='location' )
    plt.title('difference in logerror across counties')
    plt.show()
    
def county_log1(train):
    '''location compared to log error '''
    plt.figure(figsize=(17,10))
    sns.violinplot(data=train,x='location', y='Log_error',hue='location' )
    plt.title('Difference in logerror across counties')
    plt.show()
    
def county_log2(train):
    '''orange has the highest logerror out of all of them aka countys'''
    plt.figure(figsize=(17,10))
    sns.barplot(data=train,x='location', y='Log_error',hue='location' )
    plt.title('Difference in logerror across counties')
    plt.show()
    
def county_log3(train):
    '''orange has the highest logerror out of all of them aka countys'''
    plt.figure(figsize=(17,10))
    sns.scatterplot(data=train, x="Log_error", y="Log_error",hue='location')
    sns.rugplot(data=train, x="Log_error", y="Log_error",hue='location')
    plt.title('Difference in logerror across counties')
    plt.show()

    
#bedrooms graph

def bedrooms_log(train):
    '''log error compared to bedrooms'''
    brush = alt.selection(type='interval')
    alt.data_transformers.disable_max_rows()
    points = alt.Chart(train).mark_point().encode(
    x='Bathrooms',
    y='TotalRooms',
    color=alt.condition(brush, 'Log_error', alt.value('lightgray'))).add_selection(brush)
    bars = alt.Chart(train).mark_bar().encode(
    y='Log_error',
    color='location',
    x='Bathrooms').transform_filter(brush)
    return points & bars 

def bedrooms_log1(train):
    '''log error compared to bedrooms'''
    plt.figure(figsize=(17,10))
    sns.violinplot(data=train,x='Bedrooms',y='Log_error',hue='location')
    plt.title('Difference in logerror for Bedrooms')
    plt.show()
    
#bathroom graph
    
def bathrooms_log(train):
    '''log error compared to bathrooms'''
    plt.figure(figsize=(17,10))
    sns.barplot(data=train,x='Bathrooms', y='Log_error',hue='location' )
    plt.title('Difference in logerror for bathrooms')
    plt.show()
    
def bathrooms_log1(train):
    '''log error compared to bathrooms'''
    plt.figure(figsize=(17,10))
    sns.violinplot(data=train,x='Bathrooms', y='Log_error',hue='location' )
    plt.title('Difference in logerror for bathrooms')
    plt.show()
    
#squarefeet graph
    
def Squarefeet_log(train):
    '''squarefeet compared to log error '''
    plt.figure(figsize=(17,10))
    sns.scatterplot(data=train,x='Squarefeet', y='Log_error',hue='location' )
    plt.title('Difference in logerror for Squarefeet')
    plt.show()
    
def squarefeet_log1(train):
    '''squarefeet compared to log error '''
    brush = alt.selection(type='interval')
    alt.data_transformers.disable_max_rows()
    points = alt.Chart(train).mark_point().encode(
    x='Squarefeet',
    y='Log_error',
    color=alt.condition(brush, 'Log_error', alt.value('lightgray'))).add_selection(brush)
    bars = alt.Chart(train).mark_bar().encode(
    y='Log_error',
    color='location',
    x='Squarefeet').transform_filter(brush)
    return points & bars

#stats.-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def location_stats(train):
    '''t test comparring los angeles and orange county'''
    print(stats.levene(train[train.location == 'los_angeles'].Log_error, train[train.location == 'orange'].Log_error))
    print(stats.ttest_ind(train[train.location == 'los_angeles'].Log_error, train[train.location == 'orange'].Log_error, equal_var=False))
    
def location_stats1(train):
    '''t test comparring ventura and orange county'''
    print(stats.levene(train[train.location == 'ventura'].Log_error, train[train.location == 'orange'].Log_error))
    print(stats.ttest_ind(train[train.location == 'ventura'].Log_error, train[train.location == 'orange'].Log_error, equal_var=False))
    
def bedrooms_stats(train):
    '''chi test comparing bedroom and log error'''
    x = train.Bedrooms
    y = train.Log_error

    alternative_hypothesis = 'bedroom is related to log error'
    alpha = .05

    corr, p = stats.pearsonr(x, y)

    corr, p

    if p < alpha:
        print("We reject the null hypothesis")
        print("We can say that we have confidence that", alternative_hypothesis)
    else:
        print("We fail to reject the null")
        
def bathrooms_stats(train):
    '''chi test comparing bedroom and log error'''
    x = train.Bathrooms
    y = train.Log_error

    alternative_hypothesis = 'bathrooms is related to log error'
    alpha = .05
    
    corr, p = stats.pearsonr(x, y)

    corr, p

    if p < alpha:
        print("We reject the null hypothesis")
        print("We can say that we have confidence that", alternative_hypothesis)
    else:
        print("We fail to reject the null")
    
def squarefeet_stats(train):
    '''chi test comparing bedroom and log error'''
    x = train.Squarefeet
    y = train.Log_error

    alternative_hypothesis = 'squarefeet is related to logerror'
    alpha = .05

    corr, p = stats.pearsonr(x, y)

    corr, p

    if p < alpha:
        print("We reject the null hypothesis")
        print("We can say that we have confidence that", alternative_hypothesis)
    else:
        print("We fail to reject the null")
    
    p
    
#clustering-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from env import host, user, password
import acquire
import wrangle
import prepare
from sklearn.neighbors import KDTree
from scipy import stats

import warnings
warnings.filterwarnings("ignore")



def elbow_method(list_of_3_variables):
    cluster_vars = list_of_3_variables

    ks = range(2,15)
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(cluster_vars)

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_)

    print(pd.DataFrame(dict(k=ks, sse=sse)))

    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k')
    plt.show()

def elbow(df, points=10):
    ks = range(1,points+1)
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=123)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    print(pd.DataFrame(dict(k=ks, sse=sse)))
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k')
    plt.show()

def k_cluster_all(df, x, n):
    """
    Takes a dataframe and a single feature, and performs a 2d kmeans clustering on that feature against all other features in the dataframe. Also, specify the number of clusters to explore.
    """  
    kmeans = KMeans(n_clusters=n, random_state=123)
    kmeans.fit(df)
    df["cluster"] = kmeans.predict(df)
    df.cluster = 'cluster_' + (df.cluster + 1).astype('str')

    for col in df.columns:
        if col != x and col != "cluster":
            sns.relplot(data=df, x=x, y=col, hue='cluster', alpha=.3)
            plt.show()
    df.drop(columns="cluster", inplace=True)

def k_cluster_2d(df, x, y, n_max, n_min=2):
    """
    Plots a 2D cluster map of an inputted x and y, starting at 2 clusters, up to inputted max cluster amount
    Import whole dataframe, select the x and y values to cluster.
    """
    for n in range(n_min,n_max+1):
        kmeans = KMeans(n_clusters=n, random_state=123)
        kmeans.fit(df)
        df["cluster"] = kmeans.predict(df)
        df.cluster = 'cluster_' + (df.cluster + 1).astype('str')

        sns.relplot(data=df, x=x, y=y, hue='cluster', alpha=.5)
        plt.title(f'{n} Clusters')
        df.drop(columns="cluster", inplace=True)

def k_cluster_3d(df, x, y, z, n):
    """
    Displays 3d plot of clusters.
    """
    kmeans = KMeans(n_clusters=n, random_state=123)
    kmeans.fit(df)
    cluster_label = kmeans.labels_

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')
  
    scatter = ax.scatter(df[x], df[y], df[z], c=cluster_label,alpha=.5)
    legend = ax.legend(*scatter.legend_elements(),loc="lower left", title="Clusters")
    
    ax.add_artist(legend)
    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    ax.set(xlabel=x, ylabel=y, zlabel=z)
    ax.xaxis.labelpad=-5
    ax.yaxis.labelpad=-5
    ax.zaxis.labelpad=-5
    plt.show()

def get_pde(df,bw):
    """
    Assits in plotting a parcel density estimation 2d scatter plot. Use the longitude and latitude as x, y coordinates and color these points by their density.
    """
    x = df['longitude'].values
    y = df['latitude'].values
    xy = np.vstack([x,y])
    X = np.transpose(xy)
    tree = KDTree(X,leaf_size = 20 )     
    parcelDensity = tree.kernel_density(X, h=bw,kernel='gaussian',rtol=0.00001)
    return parcelDensity

def get_clusters_and_counts(k, var_list, cluster_col_name, train_scaled, test_scaled):
    """
    be sure your scaled X dataframes are named: train_scaled and test_scaled
    takes in k, list of vars to cluster on, and the new cluster id column name
    and returns the kmeans fitted object, dataframe of the train clusters with their observations, 
    test clusters and their observations, and a df of the number of observations per cluster on train. 
    """
    
    # find clusters
    kmeans = KMeans(n_clusters=k, random_state = 447)
    train_cluster_array = kmeans.fit_predict(train_scaled[var_list])
    test_cluster_array = kmeans.predict(test_scaled[var_list])
    
    # create df of cluster id with each observation
    train_clusters = pd.DataFrame(train_cluster_array, columns = [cluster_col_name], index = train_scaled.index)
    test_clusters = pd.DataFrame(test_cluster_array, columns = [cluster_col_name], index = test_scaled.index)
    
    # output number of observations in each cluster
    cluster_counts = train_clusters[cluster_col_name].value_counts()
    
    return kmeans, train_clusters, test_clusters, cluster_counts


def append_clusters_and_centroids(X_train, train_scaled, train_clusters, 
                                X_test, test_scaled, test_clusters, 
                                cluster_col_name, centroid_col_names_list, kmeans):

    """
    be sure your dataframes are named: X_train, X_test, train_scaled, test_scaled (dataframes of X scaled)
    takes in list of vars to cluster on, 
    and the new cluster id column name
    and returns the kmeans fitted object, dataframe of the train clusters with their observations, 
    test clusters and their observations, and a df of the number of observations per cluster on train. 
    """
    
    # join the cluster ID's with the X dataframes (the scaled and unscaled, train and test
    
    X_train_scaled = pd.concat([X_train, train_clusters], axis = 1)
    train_scaled = pd.concat([train_scaled, train_clusters], axis = 1)

    X_test_scaled = pd.concat([X_test, test_clusters], axis = 1)
    test_scaled = pd.concat([test_scaled, test_clusters], axis = 1)
      
    # get the centroids for  distinct cluster...
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=centroid_col_names_list).reset_index()
    centroids.rename(columns = {'index': cluster_col_name}, inplace = True)
    
    # merge the centroids with the X dataframes (both the scaled and unscaled)
    X_train_scaled = X_train_scaled.merge(centroids, how='left', on=cluster_col_name).set_index(X_train_scaled.index)
    train_scaled = train_scaled.merge(centroids, how = 'left', on = cluster_col_name).set_index(train_scaled.index)
    
    X_test_scaled = X_test_scaled.merge(centroids, how = 'left', on = cluster_col_name).set_index(X_test_scaled.index)
    test_scaled = test_scaled.merge(centroids, how = 'left', on = cluster_col_name).set_index(test_scaled.index)
    
    return X_train_scaled, train_scaled, X_test_scaled, test_scaled, centroids

def test_sig(cluster_column,df):
    """
    Takes a column of clusters and performs a t-test with the logerrors of cluster (subset) against the population logerror.
    """  
    ttest_list = []
    pval_list = []
    stat_sig = []

    for cluster in cluster_column.unique():
        ttest, pval = stats.ttest_1samp(df["Log_error"][cluster_column == cluster],df["Log_error"].mean(),axis=0,nan_policy="propagate")
        ttest_list.append(ttest)
        pval_list.append(pval)
        sig = pval < 0.05
        stat_sig.append(sig)
        
    stats_cluster_column = pd.DataFrame({"ttest":ttest_list,"pval":pval_list,"stat_sig":stat_sig})
    

