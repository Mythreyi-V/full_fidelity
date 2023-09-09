import pandas as pd
import numpy as np

from DatasetManager import DatasetManager

from sklearn.metrics import mean_absolute_percentage_error, accuracy_score, mean_squared_error, r2_score
import sklearn

import os
import joblib

import sys

import warnings
warnings.filterwarnings('ignore')

import scipy

import shap
import lime
import learning
import pyAgrum
#from acv_explainers import ACXplainer

#from anchor import anchor_tabular

import json

from tqdm import tqdm, tqdm_notebook

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from collections import Counter
import seaborn as sns

import multiprocessing as mp


def get_tree_features(cls, instance):
    tree = cls.tree_
    lvl = 0
    left_child = tree.children_left[lvl]
    right_child = tree.children_right[lvl]

    feats = []
    
    while left_child != sklearn.tree._tree.TREE_LEAF and right_child != sklearn.tree._tree.TREE_LEAF:
        feature = tree.feature[lvl]
        feats.append(feature)
        
        if instance[feature] < tree.threshold[lvl]:
            lvl = left_child
        else:
            lvl = right_child
            
        left_child = tree.children_left[lvl]
        right_child = tree.children_right[lvl]
            
            
    feat_pos = np.zeros(len(instance))
    n = len(feats)
    for i in feats:
        feat_pos[i]+=n
#        feat_pos[i]+=1
        n=n-1
    #feat_pos = set(feats)
    
    return feat_pos

def get_reg_features(cls):

    og_coef = cls.coef_
    if len(og_coef.shape) > 1:
        og_coef = og_coef[0]
    
    coef = [abs(val) for val in og_coef]
    
#     bins = pd.cut(coef, 4, retbins = True, duplicates = "drop")
#     q1_min = bins[1][-2]
    
#     feat_pos = [i for i in range(len(coef)) if coef[i] > q1_min]
    
    return coef

def get_nb_features(cls, instance):
    pred = cls.predict(instance.reshape(1, -1))
    means = cls.theta_[pred][0]
    std = np.sqrt(cls.var_[pred])[0]
    
    alt = 1-pred
    alt_means = cls.theta_[alt][0]
    alt_std = np.sqrt(cls.var_[alt])[0]

    likelihoods = []
    
    for i in range(len(means)):
        lk = scipy.stats.norm(means[i], std[i]).logpdf(instance[i])
        alt_lk = scipy.stats.norm(alt_means[i], alt_std[i]).logpdf(instance[i])
        lkhood = lk-alt_lk
        likelihoods.append(lkhood)
    
    return np.abs(likelihoods)

def get_true_rankings(cls, instance, cls_method, X_train, feat_list):
    if cls_method == "decision_tree":
        feat_pos = get_tree_features(cls, instance)
        
    elif cls_method == "logit" or cls_method == "lin_reg":
        feat_pos = get_reg_features(cls)
        
    elif cls_method == "nb":
        feat_pos = get_nb_features(cls, instance)
        
    return feat_pos

def permute_instance(instance, i, perm_iter = 100, min_i = [0], max_i=[1], mean_i=[0], mode="permutation"):
            
    permutations = np.array([instance]*perm_iter).transpose()
    
    for j in range(len(i)):
        if mode=="baseline_max":
            n_val = [max_i[j]]*perm_iter
        elif mode=="baseline" or mode=="baseline_mean":
            n_val = [mean_i[j]]*perm_iter
        elif mode=="baseline_min":
            n_val = [min_i[j]]*perm_iter
        elif mode=="baseline_0":
            n_val = [0]*perm_iter
        else:
            n_val = np.random.uniform(min_i[j], max_i[j], perm_iter)


        permutations[i[j]] = n_val
    
    permutations = permutations.transpose()

    return permutations

def cycle_values(instance, i, perm_iter = 100, min_i = [0], max_i=[1], mean_i=[0], unique_values=[[0,1]], mode="permutation"):

    permutations = np.array([instance]*perm_iter).transpose()

    for j in range(len(i)):
        if mode=="baseline_max":
            n_val = [max_i[j]]*perm_iter
        elif mode=="baseline" or mode=="baseline_mean":
            n_val = [mean_i[j]]*perm_iter
        elif mode=="baseline_min":
            n_val = [min_i[j]]*perm_iter
        elif mode=="baseline_0":
            n_val = [0]*perm_iter
        else:
            n_val = np.random.choice(unique_values[j], perm_iter)

        permutations[i[j]] = n_val
        
    permutations = permutations.transpose()

    return permutations

def permute_multiple(instance, i, columns, col_dict, perm_iter=100, min_i = [0], max_i=[1], mean_i=[0],unique_values=[[0,1]], mode="permutation"):
    
    cats = []
    nums = []
    
    if col_dict["discrete"]!=None:
        cats = [int(col) for col in i if columns[col] in col_dict["discrete"]]
    if col_dict["continuous"]!=None:
        nums = [int(col) for col in i if columns[col] in col_dict["continuous"]]
    
    cat_permutations = False
    num_permutations = False
    
    if len(cats)>0:
        mins = min_i[columns[cats]].values
        maxes = max_i[columns[cats]].values
        means = mean_i[columns[cats]].values
        uniques = unique_values[columns[cats]].values
        cat_permutations = cycle_values(instance, cats, perm_iter, mins, maxes, 
                                  means, uniques, mode) 
    if len(nums)>0:
        mins = min_i[columns[nums]].values
        maxes = max_i[columns[nums]].values
        means = mean_i[columns[nums]].values
        num_permutations = permute_instance(instance, nums, perm_iter, mins, maxes, 
                                  means, mode)
        
    permutations = np.array([instance]*perm_iter).transpose()
    if type(cat_permutations)!=bool:
        cat_permutations = cat_permutations.transpose()
        for j in cats:
            permutations[j] = cat_permutations[j]
            
    if type(num_permutations)!=bool:
        num_permutations = num_permutations.transpose()
        for j in nums:
            permutations[j] = num_permutations[j]
    permutations = permutations.transpose()
    
    return permutations

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta


def evaluate(i):
    instance = sample_instances[i]

    tr = get_true_rankings(cls, instance, cls_method, trainingdata, feat_list)

    if classification:
        pred = cls.predict(instance.reshape(1, -1))
        proba = cls.predict_proba(instance.reshape(1, -1)).reshape(2)[pred]
        p1_list = list(proba)*perm_iter

    perm_mape = np.zeros(len(instance))
    perm_rmse = np.zeros(len(instance))
    perm_r2 = np.zeros(len(instance))

    for j in range(len(instance)):
        if cats != None:
            if trainingdata.columns[j] in cat_cols:
                permutations = cycle_values(instance, [j], perm_iter, [min_X[j]], [max_X[j]], [mean_X[j]], [unique_values[trainingdata.columns[j]]], mode)
            else:
                permutations = permute_instance(instance, [j], perm_iter, [min_X[j]], [max_X[j]], [mean_X[j]], mode)
        else:
            permutations = permute_instance(instance, [j], perm_iter, [min_X[j]], [max_X[j]], [mean_X[j]], mode)

        if classification:
            p2_list = cls.predict_proba(permutations).transpose()[pred].reshape(perm_iter)
            perm_mape[j] = mean_absolute_percentage_error(p1_list, p2_list)
            perm_rmse[j] = mean_squared_error(p1_list, p2_list, squared=False)
            perm_r2[j] = r2_score(p1_list, p2_list)

    #print("Final MAPE for all features:", perm_mape)
    mape_corr = scipy.stats.kendalltau(tr, perm_mape, variant="b")[0]
    rmse_corr = scipy.stats.kendalltau(tr, perm_rmse, variant="b")[0]
    r2_corr = scipy.stats.kendalltau(tr, perm_r2, variant="b")[0]

    return mape_corr, rmse_corr, r2_corr

# path to project folder
# please change to your own
PATH = os.getcwd()

dataset_ref = sys.argv[1]
params_dir = PATH + "params"
results_dir = "results"
bucket_method = sys.argv[2]
cls_encoding = sys.argv[3]
cls_method = sys.argv[4]

classification=True

gap = 1
n_iter = 1

method_name = "%s_%s"%(bucket_method, cls_encoding)
save_to = os.path.join(PATH, dataset_ref, cls_method, method_name)

modes = ["permutation", "baseline_min", "baseline_mean", "baseline_max", "baseline_0"]

sample_size = 2
exp_iter = 10
max_feat = 10
max_prefix = 20
random_state = 22
perm_iter = 1000

dataset_ref_to_datasets = {
    "bpic2012" : ["bpic2012_accepted"],
    "sepsis_cases": ["sepsis_cases_1"],
    "production" : ["production"],
    "bpic2011": ["bpic2011_f1"],
    "hospital": ["hospital_billing_2"],
    "traffic": ["traffic_fines_1"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]

dt_train_prefixes = pd.read_csv(os.path.join(PATH, "%s/datasets/train_prefixes.csv" % (dataset_ref)))
dt_test_prefixes = pd.read_csv(os.path.join(PATH, "%s/datasets/test_prefixes.csv" % (dataset_ref)))
dt_val_prefixes = pd.read_csv(os.path.join(PATH, "%s/datasets/val_prefixes.csv" % (dataset_ref)))

dt_train_prefixes = pd.concat([dt_train_prefixes, dt_val_prefixes])

dataset_manager = DatasetManager(datasets[0])
data = dataset_manager.read_dataset()

non_num_cols = dataset_manager.dynamic_cat_cols+dataset_manager.static_cat_cols
num_cols = dataset_manager.dynamic_num_cols+dataset_manager.static_num_cols

num_data = dt_train_prefixes[num_cols]
cat_data = dt_train_prefixes[non_num_cols]


all_results = []

for dataset_name in datasets:
        
    for ii in range(n_iter):
        num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s/%s/pipelines'% 
                                                                    (dataset_ref, cls_method, method_name)))])
        dataset_manager = DatasetManager(dataset_name)
        

        for bucket in tqdm_notebook(range(num_buckets)):
            bucketID = bucket+1
            print ('Bucket', bucketID)

            #import everything needed to sort and predict
            pipeline_path = os.path.join(PATH, save_to, "pipelines/pipeline_bucket_%s.joblib" % 
                                         (bucketID))
            pipeline = joblib.load(pipeline_path)
            feature_combiner = pipeline['encoder']
            if 'scaler' in pipeline.named_steps:
                scaler = pipeline['scaler']
            else:
                scaler = None
            cls = pipeline['cls']

            #import training data for bucket
            trainingdata = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/train_data_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID)))
            targets = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/y_train_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID))).values
            
            #Identify feature names
            feat_list = [feat.replace(" ", "_") for feat in feature_combiner.get_feature_names()]
            cats = [feat for col in dataset_manager.dynamic_cat_cols+dataset_manager.static_cat_cols 
                for feat in range(len(feat_list)) if col in feat_list[feat]]
            cat_cols = trainingdata.columns[cats]
            
            #scale data if necessary
            if scaler != None:
                trainingdata = scaler.transform(trainingdata)
                trainingdata = pd.DataFrame(trainingdata, columns=feat_list)

            #find relevant samples for bucket
            sample_instances = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/test_sample_bucket_%s.csv" % 
                                      (dataset_ref, cls_method, method_name, bucketID))).values
            results = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/results_bucket_%s.csv" % 
                                      (dataset_ref, cls_method, method_name, bucketID)))

            #scale data if necessary
            if scaler != None:
                sample_instances = scaler.transform(sample_instances)
            

            #trainingdata = pd.DataFrame(trainingdata, columns=feat_list)
            min_X = np.min(trainingdata)
            max_X = np.max(trainingdata)
            mean_X = np.mean(trainingdata, axis=0)
            unique_values = pd.Series({col: trainingdata[col].unique() for col in trainingdata.columns})

            for mode in modes:
                ktb_list = []
                true_v_mape = []
                true_v_rmse = []
                true_v_r2 = []
                
                pool = mp.Pool(mp.cpu_count(), initargs=(mp.RLock(),), initializer=tqdm.set_lock)

                for result in tqdm(pool.imap(evaluate, [i for i in range(len(sample_instances))]), total = len(sample_instances)):
                    true_v_mape.append(result[0])
                    true_v_rmse.append(result[1])
                    true_v_r2.append(result[2])
                    
                results["MAPE Correctness"] = true_v_mape
                results["RMSE Correctness"] = true_v_rmse
                results["R2 Correctness"] = true_v_r2
                results["Mode"] = [mode]*results.shape[0]
                
                print(mode, "Mean correlation:", np.mean(results["R2 Correctness"]))
                
                results.to_csv(os.path.join(save_to,"samples", mode+"_results_bucket_%s.csv" % (bucketID)), 
                               index = False, sep = ";")
                
                all_results.append(results)
                
#pd.concat(all_results).to_csv(os.path.join(PATH,"%s/%s/%s/samples/all_perm_results.csv") % (dataset_ref, cls_method, method_name), 
#                               sep=";", index=False)