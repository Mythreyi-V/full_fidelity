## HYPERPARAMETER OPTIMISATION FOR PPA MODELS

import sys
import os
PATH = os.getcwd()
sys.path.append(PATH)

import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample


import time
import os
import sys
from sys import argv
import pickle
from collections import defaultdict

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

def create_and_evaluate_model(args):    
    print(args)
    global trial_nr
    trial_nr += 1
    
    start = time.time()
    score = 0
    for cv_iter in range(n_splits):
        
        dt_test_prefixes = dt_prefixes[cv_iter]
        dt_train_prefixes = pd.DataFrame()
        for cv_train_iter in range(n_splits): 
            if cv_train_iter != cv_iter:
                dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0, sort=False)
                
        #Balance unbalanced data
        if balanced_data == False:
            y = dataset_manager.get_label_numeric(dt_train_prefixes)
            case_ids = dataset_manager.get_case_ids(dt_train_prefixes)

            neg_cases = [case_ids[i] for i in range(len(case_ids)) if y[i] == 0]
            pos_cases = [case_ids[i] for i in range(len(case_ids)) if y[i] == 1]

            if len(neg_cases) > len(pos_cases):
                neg_cases = resample(neg_cases, replace = False, n_samples = len(pos_cases))
            elif len(neg_cases) < len(pos_cases):
                pos_cases = resample(pos_cases, replace = False, n_samples = len(neg_cases))

            bal_data = dt_train_prefixes.loc[dt_train_prefixes[dataset_manager.case_id_col].isin(neg_cases)]
            bal_data = bal_data.append(dt_train_prefixes.loc[dt_train_prefixes[dataset_manager.case_id_col].isin(pos_cases)])
            dt_train_prefixes = bal_data
                
        dt_test_bucket = dt_test_prefixes
        dt_train_bucket = dt_train_prefixes
        
        test_y = dataset_manager.get_label_numeric(dt_test_bucket)
        train_y = dataset_manager.get_label_numeric(dt_train_bucket)

        if len(set(train_y)) < 2:
            preds = [train_y[0]] * len(relevant_test_cases_bucket)
        else:
            feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])

            if cls_method == "xgboost":
                cls = xgb.XGBClassifier(objective='binary:logistic',
                                        n_estimators=args['n_estimators'],
                                        learning_rate= args['learning_rate'],
                                        subsample=args['subsample'],
                                        max_depth=int(args['max_depth']),
                                        colsample_bytree=args['colsample_bytree'],
                                        min_child_weight=int(args['min_child_weight']),
                                        seed=random_state)
            elif cls_method == "logit":
                cls = LogisticRegression(C=2**args['C'],
                                         random_state=random_state)
            elif cls_method == "nb":
                cls = GaussianNB(var_smoothing=args["var_smoothing"])

            pipeline = Pipeline([('encoder', feature_combiner), ('scaler', MinMaxScaler()), ('cls', cls)])            
            pipeline.fit(dt_train_bucket, train_y)

            preds = pipeline.predict(dt_test_bucket)

        if balanced_data==True:
            acc = roc_auc_score(test_y, preds)
            score += acc
        else:
            acc = f1_score(test_y, preds)
            score += acc
        
        print('Accuracy:', acc)
        
    for k, v in args.items():
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, k, v, score / n_splits))   
    fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, "processing_time", time.time() - start, 0))   
    fout_all.flush()
    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}

dataset_ref = sys.argv[1]
params_dir = "params"
n_iter = 10
bucket_method = sys.argv[2]
cls_encoding = sys.argv[3]
cls_method = sys.argv[4]
balanced_data = False

if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"

method_name = "%s_%s"%(bucket_method, cls_encoding)

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(5,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "bpic2012" : ["bpic2012_accepted"],
    "sepsis_cases": ["sepsis_cases_1"],#, "sepsis_cases_2", "sepsis_cases_4"],
    "production": ["production"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"],
    "3d" : []
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]
print(datasets)
    
train_ratio = 0.8
n_splits = 3
random_state = 22

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))
    
for dataset_name in datasets:
    
    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    data = dataset_manager.balance_data(data)

    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True}

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    else:
        max_prefix_length = 20

    # split into training and test
    print("splitting data")
    train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
    
    # Bucketing prefixes based on control flow
    bucketer_args = {'encoding_method':bucket_encoding, 
                     'case_id_col':dataset_manager.case_id_col, 
                     'cat_cols':[dataset_manager.activity_col], 
                     'num_cols':[], 
                     'random_state':random_state}
    if bucket_method == "cluster":
        bucketer_args["n_clusters"] = args["n_clusters"]
    bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
    bucket_assignments_train = bucketer.fit_predict(train_prefixes)
    
    for bucket in set(bucket_assignments_train):
        print("Optimising %s of %s buckets" % (bucket, len(set(bucket_assignments_train))))
        
        relevant_train_cases_bucket = dataset_manager.get_indexes(train_prefixes)[bucket_assignments_train == bucket]
        dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(train_prefixes, relevant_train_cases_bucket)
    
        # prepare chunks for CV
        dt_prefixes = []
        class_ratios = []
        for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(dt_train_bucket, n_splits=n_splits):
            class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
            dt_prefixes.append(train_chunk)
        #del train
        
        # set up search space
        if cls_method == "xgboost":
            space = {'n_estimators': scope.int(hp.quniform('n_estimators', 1, 175, 5)),
                    'learning_rate': hp.uniform("learning_rate", 0, 5),
                     'subsample': hp.uniform("subsample", 0.5, 1),
                     'max_depth': scope.int(hp.quniform('max_depth', 1, 30, 1)),
                     'colsample_bytree': hp.uniform("colsample_bytree", 0, 1),
                     'min_child_weight': scope.int(hp.quniform('min_child_weight', 0, 6, 1))}
        elif cls_method == "logit":
            space = {'C': hp.uniform('C', -15, 15)}
        elif cls_method == "nb":
            space = {'var_smoothing': hp.choice('v', np.logspace(0, -9, 100))}


        # optimize parameters
        trial_nr = 1
        trials = Trials()
        fout_all = open(os.path.join(PATH, params_dir, "param_optim_all_trials_%s_%s_%s_bucket_%s.csv" % (cls_method, dataset_name, method_name, bucket)), "w")
        if "prefix" in method_name:
            fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "nr_events", "param", "value", "score"))   
        else:
            fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "param", "value", "score"))   
        best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials, verbose=True)
        fout_all.close()

        # write the best parameters
        best_params = hyperopt.space_eval(space, best)
        outfile = os.path.join(PATH, params_dir, "optimal_params_%s_%s_%s_bucket_%s.pickle" % (cls_method, dataset_name, method_name, bucket))
        # write to file
        with open(outfile, "wb") as fout:
            pickle.dump(best_params, fout)
