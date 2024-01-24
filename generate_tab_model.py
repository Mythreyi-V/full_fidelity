#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install lime
#!pip install shap
#!pip install anchor-exp
#!pip install hyperopt
#!pip install imodels

import pandas as pd
import numpy as np

import xgboost as xgb

import pickle

from imodels import BayesianRuleListClassifier
from collections import OrderedDict

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelBinarizer, StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

import os
import sys
import joblib

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import pylab as pl
from pylab import savefig
plt.style.use('seaborn-deep')

import stability as st

import statistics
import scipy as scp
import math

import lime
import lime.lime_tabular

import shap

from anchor import anchor_tabular

import time
import random


# In[2]:


# path to project folder
# please change to your own
PATH = os.getcwd()

dataset = sys.argv[1]
cls_method = sys.argv[2]

classification = True

random_state = 39
num_eval = 500
n_splits = 3
random.seed(random_state)

save_to = "%s/%s/" % (PATH, dataset)
dataset_folder = "%s/datasets/" % (save_to)


# In[3]:


#Get datasets
X_train = pd.read_csv(dataset_folder+dataset+"_Xtrain.csv", index_col=False, sep = ";")#.values
X_test = pd.read_csv(dataset_folder+dataset+"_Xtest.csv", index_col=False, sep = ";")#.values
X_validation = pd.read_csv(dataset_folder+dataset+"_Xvalidation.csv", index_col=False, sep = ";")#.values

y_train = pd.read_csv(dataset_folder+dataset+"_Ytrain.csv", index_col=False, sep = ";").values.reshape(-1)
y_test = pd.read_csv(dataset_folder+dataset+"_Ytest.csv", index_col=False, sep = ";").values.reshape(-1)
y_validation = pd.read_csv(dataset_folder+dataset+"_Yvalidation.csv", index_col=False, sep = ";").values.reshape(-1)

#cat_values = pd.read_csv(dataset_folder+"/cat_cols.csv").columns

feat_list = X_train.columns
results_template = pd.read_csv(os.path.join(dataset_folder, dataset+"_results_template.csv"), index_col=False)


# In[4]:


results_template


# In[ ]:


X_train


# In[5]:


#Set hyperparameter grid
if cls_method == "xgboost":
    space = {'learning_rate': [random.uniform(0,5) for i in range(5)],
            'subsample': [random.uniform(0.5,1) for i in range(5)],
            'max_depth': np.arange(1, 33, 6),
            'colsample_bytree': [random.uniform(0,1) for i in range(5)],
            'min_child_weight': np.arange(0,6,1)}
    fit_params = {}#{"eval_set": [(X_train.values, y_train)]}
    
elif cls_method == "decision_tree":
    space = {"splitter": ["best", "random"],
            "min_samples_split": [random.uniform(0, 1) for i in range (50)],
            "max_features": [random.uniform(0,1) for i in range (50)]}
    fit_params = {"sample_weight": None}
    
elif cls_method == "logit":
    space = {"fit_intercept": [True, False],
             "penalty": ['l1', 'l2', 'elasticnet', 'none'],
             "max_iter": [random.uniform(5,200) for i in range (50)],
             "tol": np.logspace(-4, 4, 50)}
    fit_params = {"sample_weight": None}
    
elif cls_method == "brl":
    space = {"minsupport": [random.uniform(0.1,0.9) for i in range (5)],
             "max_iter": [random.randint(10000, 50000) for i in range(3)],
             "maxcardinality": [random.randint(2, 10) for i in range((3))],
             "n_chains": [random.randint(1,7) for i in range(3)]}
    
    feature_dict = OrderedDict()
    
    for i in range(len(feat_list)):
        feature_dict[feat_list[i]] = f'X_{i}'
    #cat_features = [feature_dict[value] for value in cat_values]
    
    #X_train[cat_values] = X_train[cat_values].astype(str)
        
    fit_params = {"feature_names": feat_list}
                 #'undiscretized_features': cat_features}
    
elif cls_method == "lin_reg":
    space = {"normalize": [True, False], "positive": [True, False],
             "fit_intercept": [True, False]}
    fit_params = {"sample_weight": None}

elif cls_method == "knn":
    space = {'n_neighbors': [random.randint(1, 15) for i in range(10)],
             'weights': ['uniform', 'distance'],
             'leaf_size': [random.randint(1,30) for i in range(10)],
             'p': [1,2],
             'algorithm': ['kd_tree', 'ball_tree', 'brute']}
    fit_params = {}

elif cls_method == "nb":
    space = {'var_smoothing': np.logspace(0, -9, 100)}
    fit_params = {}


# In[6]:


#X_train['is_recid'].values


# In[7]:


#Create prediction model
if classification == True:
    if cls_method == "xgboost":
        estimator = xgb.XGBClassifier(random_state = random_state)
    elif cls_method == "decision_tree":
        space["criterion"] = ["gini", "entropy"]
        estimator = DecisionTreeClassifier(random_state = random_state)
    elif cls_method == "logit":
        estimator = LogisticRegression(random_state = random_state)
    elif cls_method == "brl":
        estimator = BayesianRuleListClassifier(random_state = random_state)
    elif cls_method == "knn":
        estimator = KNeighborsClassifier()
    elif cls_method == "nb":
        estimator = GaussianNB()
        
else:
    if cls_method == "xgboost":
        estimator = xgb.XGBRegressor(random_state = random_state)
    elif cls_method == "decision_tree":
        space["criterion"] = ["mse", "friedman_mse", "mae", "poisson"]
        estimator = DecisionTreeRegressor(random_state = random_state)
    elif cls_method == "lin_reg":
        estimator = LinearRegression()
    elif cls_method == "knn":
        estimator = KNeighborsRegressor()
        
cls = GridSearchCV(estimator, space, verbose = 10)
cls.fit(X_train.values, y_train, **fit_params)


# In[8]:


cls = cls.best_estimator_
joblib.dump(cls, save_to+cls_method+"/cls.joblib")

if cls_method == "brl":
    print(cls)


# In[9]:


test_x = pd.concat([X_test, X_validation])
test_y = np.hstack([y_test, y_validation])
if cls_method == "brl":
    y_pred = cls.predict(test_x.values, threshold = 0.5)
else:
    y_pred = cls.predict(test_x.values)

if classification == True:
    print(classification_report(test_y, y_pred))
else:
    print("RMSE:", mean_squared_error(test_y, y_pred, squared = False))
    print("MAE:", mean_absolute_error(test_y, y_pred))
    print("MAPE:", mean_absolute_percentage_error(test_y, y_pred))


# In[10]:


import sklearn
sklearn.metrics.r2_score(test_y, y_pred)


# In[11]:


if classification:
    full_test = pd.concat([test_x.reset_index(), results_template], axis = 1, join = 'inner').drop(['index'], axis = 1)
    full_test["predicted"] = y_pred
    
    grouped = full_test.groupby('predicted')
    if grouped.size().min() <= 50:
      balanced = grouped.apply(lambda x: x.sample(grouped.size().min()).reset_index(drop=True))
    else:
      balanced = grouped.apply(lambda x: x.sample(50).reset_index(drop=True))
    
    test_sample = balanced[X_test.columns]
    test_sample.reset_index(drop = True, inplace = True)
    
    results_template = balanced[results_template.columns]
    results_template.reset_index(drop = True, inplace = True)
    
    if cls_method == "brl":
        preds = cls.predict(test_sample.values, threshold = 0.5)
    else:
        preds = cls.predict(test_sample.values)
    probas = [cls.predict_proba(test_sample.values)[i][preds[i]] for i in range(len(preds))]

    results_template["Prediction"] = preds
    results_template["Prediction Probability"] = probas


# In[12]:


if classification == False:
    full_test = pd.concat([test_x.reset_index(), results_template], axis = 1, join = 'inner').drop(['index'], axis = 1)
    if len(full_test) <= 100:
      sample = full_test
    else:
      sample = full_test.sample(100).reset_index(drop=True)

    test_sample = sample[X_test.columns]
    test_sample.reset_index(drop = True, inplace = True)

    results_template = sample[results_template.columns]
    results_template.reset_index(drop = True, inplace = True)
    
    preds = cls.predict(test_sample.values)
    results_template["Prediction"] = preds


# In[13]:


results_template.to_csv(os.path.join(save_to, cls_method, "results.csv"), sep = ";", index = False)
test_sample.to_csv(os.path.join(save_to, cls_method, "test_sample.csv"), sep = ";", index = False)


# In[14]:


results_template


# In[ ]:




