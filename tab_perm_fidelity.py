##EVALUATIION OF PERTURBATION TECHNIQUES USING A TRANSPARENT MODEL
##USING EVENT LOGS.

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error, accuracy_score, mean_squared_error, r2_score
import sklearn
from sklearn.utils.validation import check_symmetric
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd

import os
import joblib

import warnings
warnings.filterwarnings('ignore')

import scipy
import json

from tqdm import tqdm, tqdm_notebook

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from collections import Counter
import seaborn as sns

from itertools import filterfalse

#Extract ranks from decision tree
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
        n=n-1
    
    return feat_pos

#Extract coefficients from logistic regression model
def get_reg_features(cls):

    og_coef = cls.coef_
    if len(og_coef.shape) > 1:
        og_coef = og_coef[0]
    
    coef = [abs(val) for val in og_coef]
        
    return coef

#Rank features in Naive Bayes model
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

#Get rankings from models
def get_true_rankings(cls, instance, cls_method, X_train, feat_list):
    if cls_method == "decision_tree":
        feat_pos = get_tree_features(cls, instance)
        
    elif cls_method == "logit" or cls_method == "lin_reg":
        feat_pos = get_reg_features(cls)
        
    elif cls_method == "nb":
        feat_pos = get_nb_features(cls, instance)
        
    return feat_pos

#Perturb a continuous feature
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

#Perturb a discrete feature
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

#Evaluate permutation for a data point
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


# path to project folder
# please change to your own
PATH = os.getcwd()

dataset = "nursery"
cls_method = "nb"

classification = True

modes = ["permutation", "baseline_min", "baseline_mean", "baseline_max", "baseline_0"]

random_state = 22
exp_iter = 10
perm_iter = 1000

save_to = "%s/%s/" % (PATH, dataset)
dataset_folder = "%s/datasets/" % (save_to)
final_folder = "%s/%s/" % (save_to, cls_method)

#Get datasets
X_train = pd.read_csv(dataset_folder+dataset+"_Xtrain.csv", index_col=False, sep = ";")
y_train = pd.read_csv(dataset_folder+dataset+"_Ytrain.csv", index_col=False, sep = ";")
test_x = pd.read_csv(final_folder+"test_sample.csv", index_col=False, sep = ";")
results = pd.read_csv(os.path.join(final_folder,"results.csv"), index_col=False, sep = ";")
actual = results["Actual"].values

with open(dataset_folder+"col_dict.json", "r") as f:
    col_dict = json.load(f)
f.close()

feat_list = [each.replace(' ','_') for each in X_train.columns]

cls = joblib.load(save_to+cls_method+"/cls.joblib")
scaler = joblib.load(save_to+"/scaler.joblib")

#Get relevant values for permutation from all columns
min_X = np.min(X_train)
max_X = np.max(X_train)
mean_X = np.mean(X_train, axis=0)
unique_values = pd.Series({col: X_train[col].unique() for col in X_train.columns})

#Permute with each of the possible ways
for mode in modes:
    print(mode)
    ktb_list = []
    true_v_mape = []
    true_v_rmse = []
    true_v_r2 = []

    for i in tqdm(range(len(test_x.values))):
        instance = test_x.values[i]

        tr = get_true_rankings(cls, instance, cls_method, X_train, feat_list)

        if classification:
            pred = cls.predict(instance.reshape(1, -1))
            proba = cls.predict_proba(instance.reshape(1, -1)).reshape(2)[pred]
            p1_list = list(proba)*perm_iter

        perm_mape = np.zeros(len(instance))
        perm_rmse = np.zeros(len(instance))
        perm_r2 = np.zeros(len(instance))
        
        #Perturb and check difference in output
        for j in range(len(instance)):
            if col_dict["continuous"] != None:
                if X_train.columns[j] in col_dict["continuous"]:
                    permutations = permute_instance(instance, [j], perm_iter, [min_X[j]], [max_X[j]], [mean_X[j]], mode)
                else:
                    permutations = cycle_values(instance, [j], perm_iter, [min_X[j]], [max_X[j]], [mean_X[j]], [unique_values[X_train.columns[j]]], mode)
            else:
                permutations = cycle_values(instance, [j], perm_iter, [min_X[j]], [max_X[j]], [mean_X[j]], [unique_values[X_train.columns[j]]], mode)

            if classification:
                p2_list = cls.predict_proba(permutations).transpose()[pred].reshape(perm_iter)
                perm_mape[j] = mean_absolute_percentage_error(p1_list, p2_list)
                perm_rmse[j] = mean_squared_error(p1_list, p2_list, squared=False)
                perm_r2[j] = r2_score(p1_list, p2_list)
    
        #Calculate correlation between true and explanation rankings
        #print("Final MAPE for all features:", perm_mape)
        mape_corr = scipy.stats.kendalltau(tr, perm_mape, variant="b")[0]
        rmse_corr = scipy.stats.kendalltau(tr, perm_rmse, variant="b")[0]
        r2_corr = scipy.stats.kendalltau(tr, perm_r2, variant="b")[0]

        true_v_mape.append(mape_corr)
        true_v_rmse.append(rmse_corr)
        true_v_r2.append(r2_corr)
        
    print("MAPE", np.nanmean(true_v_mape))
    print("RMSE", np.nanmean(true_v_rmse))
    print("R-Squared", np.nanmean(true_v_r2))

    results["MAPE Correctness"] = true_v_mape
    results["RMSE Correctness"] = true_v_rmse
    results["R2 Correctness"] = true_v_r2
    results.to_csv(os.path.join(save_to, cls_method, mode+"_results.csv"), index = False, sep = ";")

