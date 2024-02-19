##GENERATE A SURROGATE MODEL USING ACV FOR EVENT LOGS

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, classification_report, roc_auc_score, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_val_score
import sklearn

import sys
import os
import joblib

import warnings
warnings.filterwarnings('ignore')

from acv_explainers import ACXplainer

import random

from tqdm import tqdm_notebook

from hyperopt import fmin, tpe, hp, Trials, rand#, early_stop
from hyperopt.pyll import scope

from DatasetManager import DatasetManager
import BucketFactory

# path to project folder
# please change to your own
PATH = os.getcwd()

dataset = sys.argv[1]
bucket_method = sys.argv[2]
encoding = sys.argv[3]
cls_method = sys.argv[4]

method_name = bucket_method+"_"+encoding

random_state = 22
exp_iter = 10

method_folder = os.path.join(PATH, dataset, cls_method, method_name)
dataset_folder = os.path.join(PATH, dataset, "datasets")

print(method_folder)
print(dataset_folder)

dataset_ref_to_datasets = {
    "bpic2012" : ["bpic2012_accepted"],
    "sepsis_cases": ["sepsis_cases_1"],
    "production" : ["production"],
    "bpic2011": ["bpic2011_f1"],
    "hospital": ["hospital_billing_2"],
    "traffic": ["traffic_fines_1"]
}


datasets = [dataset] if dataset not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset]

num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s/%s/pipelines'% (dataset, cls_method, method_name)))])

for dataset_name in datasets:
    #Set up original datasets and models
    dataset_manager = DatasetManager(dataset_name)
    
    min_prefix_length = 1
    max_prefix_length = num_buckets

    dt_train_prefixes = pd.read_csv(os.path.join(dataset_folder, "train_prefixes.csv"))
    dt_train_prefixes = dataset_manager.generate_prefix_data(dt_train_prefixes, min_prefix_length, max_prefix_length)

    dt_val_prefixes = pd.read_csv(os.path.join(dataset_folder, "val_prefixes.csv"))
    dt_val_prefixes = dataset_manager.generate_prefix_data(dt_val_prefixes, min_prefix_length, max_prefix_length)
    
    dt_test_prefixes = pd.read_csv(os.path.join(dataset_folder, "test_prefixes.csv"))
    dt_test_prefixes = dataset_manager.generate_prefix_data(dt_test_prefixes, min_prefix_length, max_prefix_length)
    
    if bucket_method == "state":
        bucket_encoding = "last"
    else:
        bucket_encoding = "agg"
    
    bucketer_args = {'encoding_method':bucket_encoding,
                     'case_id_col':dataset_manager.case_id_col, 
                     'cat_cols':[dataset_manager.activity_col], 
                     'num_cols':[], 
                     'random_state':random_state}
    bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)

    bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
    bucket_assignments_val = bucketer.predict(dt_val_prefixes)
    bucket_assignments_test = bucketer.predict(dt_test_prefixes)
    
for bucket in tqdm_notebook(range(num_buckets)):
    bucketID = bucket+1
    print ('Bucket', bucketID)

    #import everything needed to sort and predict
    pipeline_path = os.path.join(method_folder, "pipelines/pipeline_bucket_%s.joblib" % 
                                 (bucketID))
    pipeline = joblib.load(pipeline_path)
    feature_combiner = pipeline['encoder']
    if 'scaler' in pipeline.named_steps:
        scaler = pipeline['scaler']
    else:
        scaler = None
    cls = pipeline['cls']

    relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucketID]
    dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_train_cases_bucket)

    X_train = feature_combiner.transform(dt_train_bucket)
    if scaler!=None:
        X_train = scaler.transform(X_train)
        
    relevant_val_cases_bucket = dataset_manager.get_indexes(dt_val_prefixes)[bucket_assignments_val == bucketID]
    dt_val_bucket = dataset_manager.get_relevant_data_by_indexes(dt_val_prefixes, relevant_val_cases_bucket)

    X_val = feature_combiner.transform(dt_val_bucket)
    if scaler!=None:
        X_val = scaler.transform(X_val)
    
    relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucketID]
    dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)

    test_x = feature_combiner.transform(dt_test_bucket)
    if scaler!=None:
        test_x = scaler.transform(test_x)
    
    
    Y_pred = cls.predict(X_train)
    Y_val = cls.predict(X_val)
    test_pred = cls.predict(test_x)
    
    full_train_x = np.vstack((X_train, X_val))
    full_train_y = np.hstack((Y_pred, Y_val))
    
    #Set up hyperparameter optimisation
    kf = KFold(n_splits=5, shuffle = True, random_state=random_state)

    space = {"n_estimators": scope.int(hp.quniform('n_estimators', 1, 100, q=1)),
            "max_depth": scope.int(hp.quniform('max_depth', 1, 100, q=1)),
            "sample_fraction": (hp.quniform('sample_fraction', 0.0001, 1, q=0.4))}

    trials = Trials()
    
    def acv_classifier_optimisation(args, random_state = random_state, cv = kf, X = X_train, y = Y_pred,
                                   X_val = X_val, Y_val = Y_val):
        score = []

        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            estimator = ACXplainer(classifier = True, n_estimators = args["n_estimators"], 
                                   max_depth = args['max_depth'], sample_fraction = args["sample_fraction"])
            estimator.fit(X_train, y_train)

            score.append(f1_score(Y_val, estimator.predict(X_val)))
        
        score = np.mean(score)

        return -score

    best = fmin(acv_classifier_optimisation, verbose=0, space = space, algo=rand.suggest, max_evals = 50, trials=trials, 
                rstate=np.random.default_rng(random_state))#, early_stop_fn=early_stop.no_progress_loss(3))
    
    #train surrogate model with optimal hyperparameters
    explainer = ACXplainer(classifier = True, n_estimators = int(best['n_estimators']), 
                           max_depth = int(best['max_depth']), sample_fraction = best['sample_fraction'])
    explainer.fit(full_train_x, full_train_y)
    
    print("Training Score:", f1_score(cls.predict(full_train_x), explainer.predict(full_train_x)))
    print("Testing Score:", f1_score(cls.predict(test_x), explainer.predict(test_x)))
    
    #save surrogate model
    joblib.dump(explainer, method_folder+"/acv_surrogate/acv_explainer_bucket_%s.joblib"%(bucketID))
    
    
print("All surrogate models created")
