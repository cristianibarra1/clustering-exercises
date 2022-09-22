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
    pf = PolynomialFeatures(degree=1)
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

