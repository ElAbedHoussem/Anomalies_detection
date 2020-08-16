import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import datetime
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from math import sqrt
from numpy import hstack
from numpy import vstack
from numpy import asarray

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


seed = 32
CLEANED_DATA = "cleaned_data.xlsx"
cleaned_data_with_AEGMM= 'cleaned_data_with_AEGMM.csv'



def create_statistic_data(full_data):
    new_data = pd.DataFrame()
    #For each year
    full_data['DATE'] = full_data.index
    for year in pd.unique(pd.DatetimeIndex(full_data['DATE']).year):
        local_year_df = full_data[pd.DatetimeIndex(full_data['DATE']).year == year]
        #for each month
        for month in pd.unique(pd.DatetimeIndex(local_year_df['DATE']).month):
            #Get all samples for each month
            KELIBIA = full_data[(pd.DatetimeIndex(full_data['DATE']).month== month) & (pd.DatetimeIndex(full_data['DATE']).year ==year)]
            date = KELIBIA[['DATE']]
            # get min, max, mean, std, 25%, 50% and 75% for each column
            KELIBIA = KELIBIA.describe().drop(['count'],axis=0)
            #convert the DF from N*M dimonssion to 1*(M*N)  dimonssion
            KELIBIA = KELIBIA.unstack().to_frame().T
            KELIBIA.columns = ['_'.join(column) for column in KELIBIA.columns]
            #set the first date to meet as index 
            KELIBIA['DATE'] = date.iloc[0][0]
            KELIBIA = KELIBIA.set_index(KELIBIA['DATE'])

            new_data = new_data.append(KELIBIA)
    new_data = new_data.drop(['DATE','APP_std','APP_min','APP_25%','APP_50%','APP_75%','APP_max'],axis=1)
    new_data.rename(columns={'APP_mean':'APP'},inplace=True)

    return new_data  

def preprocessing(original_data):
    clean_data = original_data.copy()
    clean_data.drop_duplicates(keep='first', inplace=True)
    ### fill missing data 
    clean_data.interpolate(inplace=True)
    clean_data.fillna(method='bfill',inplace=True)    
    #print('missing values:',clean_data.isna().sum().sum())
    return clean_data


def create_x_train_x_test(ts_features_targets):
    # split data into train & test sets
    X = ts_features_targets.drop('APP', axis=1)
    y = ts_features_targets[['APP']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed, shuffle=True)
    return  X_train, X_test, y_train, y_test, X, y


# scaling and transform data
def data_scaling_transform(X_train, X_test, y_train, y_test):
    ### transform input variables
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    ## transform target variables
    transformer = PowerTransformer().fit(y_train)
    y_train = transformer.transform(y_train)
    y_test = transformer.transform(y_test)     
    return X_train, X_test, y_train, y_test


def get_models():
    models = list()
    models.append(LinearRegression())
    models.append(HuberRegressor())
    models.append(Ridge())
    models.append(Lasso(max_iter=1500))
    models.append(KNeighborsRegressor(n_neighbors=3))
    models.append(SVR())
    models.append(DecisionTreeRegressor())
    models.append(ExtraTreeRegressor())
    models.append(BaggingRegressor(base_estimator=DecisionTreeRegressor()))
    models.append(RandomForestRegressor())
    models.append(ExtraTreesRegressor())
    models.append(AdaBoostRegressor(random_state=seed))
    models.append(GradientBoostingRegressor(random_state=seed))
    models.append(XGBRegressor(objective ='reg:squarederror'))
    models.append(MLPRegressor())
    return models

# collect out of fold predictions form k-fold cross validation
def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    # define split of data
    splits = 5
    kfold = KFold(n_splits=5, shuffle=True)
    # enumerate splits
    for train_ix, test_ix in kfold.split(X):
        splits-=1
        fold_yhats = list()
        # get data
        train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
        train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
        meta_y.extend(test_y)
        # fit and make predictions with each sub-model
        for model in models:
            #print((train_X.dtypes), (train_y.dtypes))            
            model.fit(train_X, train_y)
            yhat = model.predict(test_X)
            # store columns
            fold_yhats.append(yhat.reshape(len(yhat),1))
        # store fold yhats as columns
        meta_X.append(hstack(fold_yhats))
    return vstack(meta_X), asarray(meta_y)

# fit all base models on the training dataset
def fit_base_models(X, y, models):
    for index, model in enumerate(models):
        models[index] = model.fit(X, y)
    return models   

# fit a meta model
def fit_meta_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# evaluate a list of models on a dataset
def evaluate_models(X, y, models):
    df  = pd.DataFrame()
    for model in models:
        yhat = model.predict(X)
        mse = mean_squared_error(y, yhat)
        #print('%s: RMSE: %.3f , R2: %3f '  % (model.__class__.__name__, sqrt(mse), r2_score(y, yhat)))
        df = df.append(pd.DataFrame([[model.__class__.__name__, sqrt(mse), r2_score(y, yhat)]],columns=['model_name','RMSE','R2']))
    return df

def super_learner_predictions(X, models, meta_model):
    meta_X = list()    
    for model in models:
        yhat = model.predict(X)
        meta_X.append(yhat.reshape(len(yhat),1))
    meta_X = hstack(meta_X)
    # predict
    return meta_model.predict(meta_X)
    
def train(X, X_val, y, y_val):
   
    # get models
    models = get_models()
    # get out of fold predictions
    start_time = time.time() 
    meta_X, meta_y = get_out_of_fold_predictions(X, y, models)
    #print('Meta ', len(meta_X), len(meta_y))
    # fit base models
    models = fit_base_models(X, y, models)
    # fit the meta model
    meta_model = fit_meta_model(meta_X, meta_y)
    # evaluate base models
    res = evaluate_models(X_val, y_val, models)
    # evaluate meta model
    yhat = super_learner_predictions(X_val, models, meta_model)

    #res  = pd.DataFrame()
    res = res.append(pd.DataFrame([[meta_model.__class__.__name__, sqrt(mean_squared_error(y_val, yhat)), r2_score(y_val, yhat)]],columns=['model_name','RMSE','R2']))
    #print('Super Learner: RMSE %.3f' % (sqrt(mean_squared_error(y_val, yhat))))
    #print('Super Learner: R2 %.3f' % (r2_score(y_val, yhat)))
    return res, models, meta_model

def predict(X, models, meta_model):
    yhat = super_learner_predictions(X, models, meta_model)
    return yhat

def prepare_and_train(ts_features_targets=None, cleaned_data_with_AEGMM= None):

    if cleaned_data_with_AEGMM != None:
        ts_features_targets = pd.read_csv(cleaned_data_with_AEGMM)

    ts_features_targets = ts_features_targets.set_index(ts_features_targets['DATE']).drop(['DATE'],axis = 1)
    ts_features_targets = create_statistic_data(ts_features_targets)
    ts_features_targets = preprocessing(ts_features_targets)
    X_train, X_test, y_train, y_test, X, y= create_x_train_x_test(ts_features_targets)
    X_train, X_test, y_train, y_test = data_scaling_transform(X_train, X_test, y_train, y_test)
    res, models, meta_model = train(X_train.reset_index().drop(['DATE'], axis = 1), X_test.reset_index().drop(['DATE'], axis = 1), pd.DataFrame(y_train,columns=['app']).app, pd.DataFrame(y_test,columns=['app']).app)
    return(res.R2.max())



