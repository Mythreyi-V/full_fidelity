##GENERATE A SURROGATE MODEL USING ACV FOR TABULAR DATA

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, classification_report, roc_auc_score, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import sklearn

import sys
import os
import joblib

import warnings
warnings.filterwarnings('ignore')

from acv_explainers import ACXplainer

import random

from tqdm import tqdm_notebook

from hyperopt import fmin, tpe, hp, Trials, rand, early_stop
from hyperopt.pyll import scope

# path to project folder
# please change to your own
PATH = os.getcwd()

dataset = sys.argv[1]
cls_method = sys.argv[2]

random_state = 22
exp_iter = 10

save_to = "%s/%s/" % (PATH, dataset)
dataset_folder = "%s/datasets/" % (save_to)
final_folder = "%s/%s/" % (save_to, cls_method)

#Get datasets
X_train = pd.read_csv(dataset_folder+dataset+"_Xtrain.csv", index_col=False, sep = ";")
y_train = pd.read_csv(dataset_folder+dataset+"_Ytrain.csv", index_col=False, sep = ";")
test_x = pd.read_csv(final_folder+"test_sample.csv", index_col=False, sep = ";").values
results = pd.read_csv(os.path.join(final_folder,"results.csv"), index_col=False, sep = ";")

feat_list = [each.replace(' ','_') for each in X_train.columns]

#import underlying model and scaler
cls = joblib.load(save_to+cls_method+"/cls.joblib")
scaler = joblib.load(save_to+"/scaler.joblib")

#Get model predictions for all instances
Y_pred = cls.predict(X_train.values)
test_pred = cls.predict(test_x)

#Set up hyperparameter optimisation
kf = KFold(n_splits=5, shuffle = True, random_state=random_state)

space = {"n_estimators": scope.int(hp.quniform('n_estimators', 1, 20, q=1)),
        "max_depth": scope.int(hp.quniform('max_depth', 1, 20, q=1)),
        "sample_fraction": (hp.quniform('sample_fraction', 0.0001, 1, q=0.4))}

trials = Trials()

def acv_classifier_optimisation(args, random_state = random_state, cv = kf, X = X_train.values, y = Y_pred):
    score = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        estimator = ACXplainer(classifier = True, n_estimators = args["n_estimators"], 
                               max_depth = args['max_depth'], sample_fraction = args["sample_fraction"])
        estimator.fit(X_train, y_train)

        score.append(f1_score(y_test, estimator.predict(X_test)))

    score = np.mean(score)

    return -score

best = fmin(acv_classifier_optimisation, verbose=0, space = space, algo=rand.suggest, max_evals = 50, trials=trials, 
                rstate=np.random.default_rng(random_state), early_stop_fn=early_stop.no_progress_loss(3))

#Train surrogate model using best parameters
explainer = ACXplainer(classifier = True, verbose = 0, n_estimators = int(best['n_estimators']), 
                       max_depth = int(best['max_depth']), sample_fraction = best["sample_fraction"])
explainer.fit(X_train, Y_pred)

print("Training Accuracy:", f1_score(cls.predict(X_train.values), explainer.predict(X_train)))
print("Testing Accuracy:", f1_score(cls.predict(test_x), explainer.predict(test_x)))

#save surrogate model
joblib.dump(explainer, save_to+cls_method+"/acv_explainer_test.joblib")

