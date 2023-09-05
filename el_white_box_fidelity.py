import pandas as pd
import numpy as np

from sklearn.metrics import f1_score; classification_report; roc_auc_score; r2_score; mean_absolute_percentage_error
from sklearn.model_selection import KFold
import sklearn

import os
import sys
import joblib

import warnings
warnings.filterwarnings('ignore')

import scipy

import lime
import lime.lime_tabular
import shap

from learning import *
import pyAgrum

from acv_explainers import ACXplainer

from tqdm import tqdm

import json
from collections import Counter

def get_reg_features(cls; percentile):
    
    og_coef = cls.coef_
    if len(og_coef.shape) > 1:
        og_coef = og_coef[0]
    
    coef = [abs(val) for val in og_coef]
    
    min_coef = min(coef)
    max_coef = max(coef)
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k
    
    feat_pos = [i for i in range(len(coef)) if coef[i] >= q1_min]
    
    return coef; feat_pos

def get_nb_features(cls; instance; percentile):
    pred = cls.predict(instance.reshape(1; -1))
    means = cls.theta_[pred][0]
    std = np.sqrt(cls.var_[pred])[0]

    alt = 1-pred
    alt_means = cls.theta_[alt][0]
    alt_std = np.sqrt(cls.var_[alt])[0]
    
    likelihoods = []
    
#     for i in range(len(means)):
#         lkhood = scipy.stats.norm(means[i]; std[i]).logpdf(instance[i])
#         #likelihoods.append(abs(lkhood))
#         likelihoods.append(lkhood)

    for i in range(len(means)):
        lk = scipy.stats.norm(means[i]; std[i]).logpdf(instance[i])
        alt_lk = scipy.stats.norm(alt_means[i]; alt_std[i]).logpdf(instance[i])
        lkhood = abs(lk-alt_lk)
        likelihoods.append(lkhood)
    
    min_coef = min(likelihoods)
    max_coef = max(likelihoods)
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k
    
#     bins = pd.cut(likelihoods; 10; retbins = True; duplicates = "drop")[1]
#     lim_1 = bins[-2]
#     lim_2 = bins[1]
    
#     sortedls = sorted(likelihoods; reverse=True)
#     pos = math.ceil(len(likelihoods)/4)
#     lim = likelihoods[pos]
    
    feat_pos = [i for i in range(len(likelihoods)) if likelihoods[i] >= q1_min]# or likelihoods[i] <= lim_2]
    
    return likelihoods; feat_pos

def get_tree_features(cls; instance):
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
            
            
    feat_pos = set(feats)
    
    score = np.zeros(len(instance))
    n = len(feats)
    for i in feats:
        score[i]+=n
        n=n-1
    
    return score; feat_pos

def get_path_depths(tree; feat_list; cur_depth = 0; lvl = 0; depths = []):
    
    left_child = tree.children_left[lvl]
    right_child = tree.children_right[lvl]
    
    if left_child == sklearn.tree._tree.TREE_LEAF:
        depths.append(cur_depth)
        
    else:
        depths = get_path_depths(tree; feat_list; cur_depth+1; left_child; depths)
        depths = get_path_depths(tree; feat_list; cur_depth+1; right_child; depths)
    return depths

def get_shap_features(explainer; instance; cls; classification; exp_iter; feat_list; percentile):
    
    shap_exp = []
    
    pred = cls.predict(instance.reshape(1; -1))
    
    for i in range(exp_iter):
        if type(explainer) == shap.explainers._tree.Tree:
            exp = explainer(instance; check_additivity = False).values
        else:
            exp = explainer(instance.reshape(1; -1)).values
        
        #print(exp.shape)
        #print(exp)
        
        if exp.shape == (1; len(feat_list); 2):
            exp = exp[0]
            
        #print(exp.shape)
        
        if exp.shape == (len(feat_list); 2):
            exp = np.array([feat[pred] for feat in exp]).reshape(len(feat_list))
        elif exp.shape == (1; len(feat_list)) or exp.shape == (len(feat_list); 1):
            exp = exp.reshape(len(feat_list))
            
        #print(np.array(exp).shape)
            
        shap_exp.append(exp)
        
    #print(np.array(shap_exp).shape)
        
    if np.array(shap_exp).shape != (exp_iter; len(feat_list)):
        raise Exception("Explanation shape is not correct. It is"; np.array(shap_exp).shape; "instead of the expected"; (exp_iter; len(feat_list)))
    
#     if classification==True; type(explainer) == shap.explainers._tree.Tree:
#         shap_exp = []
#         for each in full_exp:
#             single_exp = [feat[0] for feat in each]
#             shap_exp.append(single_exp)
#     else:
#         shap_exp = full_exp
        
    avg_val = np.average(shap_exp; axis = 0)
    abs_val = [abs(val) for val in avg_val]
    
    #Get recall and precision for the average of shap values
    min_coef = min(abs_val)
    max_coef = max(abs_val)
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k

    sorted_val = np.copy(abs_val)
    sorted_val.sort()
    
    shap_features = set([i for i in range(len(feat_list)) if abs_val[i] > q1_min])
    
    return abs_val; shap_features

def get_lime_features(explainer; instance; cls; classification; exp_iter; feat_list; percentile):
    lime_exp = []
    
    for i in range(exp_iter):
        if classification==True:
            lime_exp.extend(explainer.explain_instance(instance; cls.predict_proba; 
                                                num_features=len(feat_list); labels=[0;1]).as_list())
        else:
            lime_exp.extend(explainer.explain_instance(instance; cls.predict; 
                                                num_features=len(feat_list); labels=[0;1]).as_list())
            
    weights = [[] for each in feat_list]
    for exp in lime_exp:
        feat = exp[0]
        if '<' in feat:
            feat = exp[0].replace("= ";'')
            parts = feat.split('<')
        elif '>' in feat:
            feat = exp[0].replace("= ";'')
            parts = feat.split('>')
        else:
            parts = feat.split("=")
        
        for part in parts:
            if part.replace('.';'').replace(' ';'').isdigit()==False:
                feat_name = part.replace(' ';'')
        n = feat_list.index(feat_name)
        weights[n].append(exp[1])
    
    weights = np.transpose(weights)
    avg_weight = np.average(np.array(weights); axis = 0)
    abs_weight = [abs(weight) for weight in avg_weight]
    
    min_coef = min(abs_weight)
    max_coef = max(abs_weight)
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k
    
    sorted_weight = np.copy(abs_weight)
    sorted_weight.sort()
    
    lime_features = set([i for i in range(len(feat_list)) if abs_weight[i] >= q1_min])
    
    return abs_weight; lime_features

def get_linda_features(instance; cls; scaler; dataset; exp_iter; feat_list; percentile):
    label_lst = ["Negative"; "Positive"]
    
    feat_pos = []
    lkhoods = []
    
    for i in range(exp_iter):
        [bn; inference; infoBN] = generate_BN_explanations(instance; label_lst; feat_list; "Result"; 
                                                                       None; scaler; cls; save_to+"/"+cls_method+"/"; dataset; show_in_notebook = False)
        
        ie = pyAgrum.LazyPropagation(bn)
        result_posterior = ie.posterior(bn.idFromName("Result")).topandas()
        result_proba = result_posterior.loc["Result"; label_lst[instance['predictions']]]
        row = instance['original_vector']
        #print(row)

        likelihood = [0]*len(feat_list)

        for j in range(len(feat_list)):
            var_labels = bn.variable(feat_list[j]).labels()
            str_bins = list(var_labels)
            bins = []

            for disc_bin in str_bins:
                disc_bin = disc_bin.strip('"(]')
                cat = [float(val) for val in disc_bin.split(';')]
                bins.append(cat)

            for k in range(len(bins)):
                if k == 0 and row[j] <= bins[k][0]:
                    feat_bin = str_bins[k]
                elif k == len(bins)-1 and row[j] >= bins[k][1]:
                    feat_bin = str_bins[k]
                elif row[j] > bins[k][0] and row[j] <= bins[k][1]:
                    feat_bin = str_bins[k]

            ie = pyAgrum.LazyPropagation(bn)
            ie.setEvidence({feat_list[j]: feat_bin})
            ie.makeInference()
            
            result_posterior = ie.posterior(bn.idFromName("Result")).topandas()
            new_proba = result_posterior.loc["Result"; label_lst[instance['predictions']]]
            #print(result_proba; new_proba)
            proba_change = result_proba-new_proba
            likelihood[j] = abs(proba_change)

        lkhoods.append(likelihood)
        
    min_coef = min( np.mean(lkhoods; axis=0))
    max_coef = max( np.mean(lkhoods; axis=0))
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k

    #If fixing all features produces the same result for the class;
    #return all features
    if len(set(np.mean(lkhoods; axis=0)))==1:
        feat_pos.extend(range(len(feat_list)))
    else:
        feat_pos.extend(list(np.where(np.mean(lkhoods; axis=0) >= q1_min)[0]))

    feat_pos = set(feat_pos)
    
    return np.mean(lkhoods; axis=0); feat_pos

def get_acv_features(explainer; instance; cls; X_train; y_train; exp_iter):
    instance = instance.reshape(1; -1)
    y = cls.predict(instance)
    
    t=np.var(y_train)

    feats = []
    feat_imp = []

    for i in range(exp_iter):
        sufficient_expl; sdp_expl; sdp_global = explainer.sufficient_expl_rf(instance; y; X_train; y_train;
                                                                                 t=t; pi_level=0.8)
        clean_expl = sufficient_expl.copy()
        clean_expl = clean_expl[0]
        clean_expl = [sublist for sublist in clean_expl if sum(n<0 for n in sublist)==0 ]

        clean_sdp = sdp_expl[0].copy()
        clean_sdp = [sdp for sdp in clean_sdp if sdp > 0]
        
        lximp = explainer.compute_local_sdp(X_train.shape[1]; clean_expl)
        feat_imp.append(lximp)
        
        if len(clean_expl)==0 or len(clean_expl[0])==0:            
            print("No explamation meets pi level")
        else:
            lens = [len(i) for i in clean_expl]
            print(lens)
            me_loc = [i for i in range(len(lens)) if lens[i]==min(lens)]
            mse_loc = np.argmax(np.array(clean_sdp)[me_loc])
            mse = np.array(clean_expl)[me_loc][mse_loc]
            feats.extend(mse)

    if len(feats)==0:
        feat_pos = []
    else:
        feat_pos = set(feats)
    
      
    feat_imp = np.mean(feat_imp; axis=0)
    
    return feat_imp; feat_pos

def get_explanation_features(explainer; instance; cls; scaler; dataset; 
                             classification; exp_iter; xai_method; feat_list; X_train; y_train; percentile):
    if xai_method == "SHAP":
        exp_score; feat_pos = get_shap_features(explainer; instance; cls; classification; exp_iter; feat_list; percentile)
        
    elif xai_method == "LIME":
        exp_score; feat_pos = get_lime_features(explainer; instance; cls; classification; exp_iter; feat_list; percentile)
        
    elif xai_method == "LINDA":
        exp_score; feat_pos = get_linda_features(instance; cls; scaler; dataset; exp_iter; feat_list; percentile)

    elif xai_method == "ACV":
        exp_score; feat_pos = get_acv_features(explainer; instance; cls; X_train; y_train; exp_iter)
        
    explanation_features = [feat_list[i] for i in feat_pos]
    #explanation_features = set(explanation_features)
        
    return exp_score; explanation_features

def get_true_features(cls; instance; cls_method; X_train; feat_list; percentile):
    if cls_method == "decision_tree":
        true_score; feat_pos = get_tree_features(cls; instance)
        
    elif cls_method == "logit" or cls_method == "lin_reg":
        true_score; feat_pos = get_reg_features(cls; percentile)
        
    elif cls_method == "nb":
        true_score; feat_pos = get_nb_features(cls; instance; percentile)
        
    true_features = [feat_list[i] for i in feat_pos]
    true_features = set(true_features)
    
    #print(feat_pos)
    
    return true_score; true_features

# path to project folder
# please change to your own
PATH = os.getcwd()

print(sys.argv)

dataset = sys.argv[1]
cls_method = sys.argv[2]
xai_method = sys.argv[3]
classification = False if sys.argv[4] == "0" else True

random_state = 22
exp_iter = 5
percentile = 0.05

save_to = "%s/%s/" % (PATH; dataset)
dataset_folder = "%s/datasets/" % (save_to)
final_folder = "%s/%s/" % (save_to; cls_method)

#Get datasets
X_train = pd.read_csv(dataset_folder+dataset+"_Xtrain.csv"; index_col=False; sep = ";")
y_train = pd.read_csv(dataset_folder+dataset+"_Ytrain.csv"; index_col=False; sep = ";")
test_x = pd.read_csv(final_folder+"test_sample.csv"; index_col=False; sep = ";").values
results = pd.read_csv(os.path.join(final_folder;"results.csv"); index_col=False; sep = ";")
actual = results["Actual"].values

with open(dataset_folder+"col_dict.json"; "r") as f:
    col_dict = json.load(f)
f.close()

feat_list = [each.replace(' ';'_') for each in X_train.columns]

cls = joblib.load(save_to+cls_method+"/cls.joblib")
scaler = joblib.load(save_to+"/scaler.joblib")

if xai_method == "SHAP":
    if cls_method == "xgboost" or cls_method == "decision_tree":
        explainer = shap.Explainer(cls)
    elif cls_method == "nb":
        if classification:
            explainer = shap.Explainer(cls.predict_proba; X_train)
        else:
            raise TypeError("Cannot run naive bayes with regression")
    else:
        explainer = shap.Explainer(cls; X_train)
        
elif xai_method == "LIME":
    if col_dict['discrete'] != None:
        cat_cols = [each.replace(' ';'_') for each in col_dict['discrete']]
        col_inds = [feat_list.index(each) for each in cat_cols]
    else:
        col_inds = []
    
    if classification==True:
        class_names=['Negative';'Positive']# negative is 0; positive is 1; 0 is left; 1 is right
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values; feature_names = feat_list; 
                                                            class_names=class_names; categorical_features = col_inds;
                                                            discretize_continuous=True)
    else:
        class_names = ['Final Value']
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values; feature_names = feat_list; 
                                                           class_names=class_names; discretize_continuous=True; 
                                                           categorical_features = col_inds; mode = "regression")
                
elif xai_method == "LINDA":
    test_dict = generate_local_predictions( test_x; results["Actual"].values; cls; scaler; None )
#    feat_list = feat_list+["Result"]

    explainer = None

elif xai_method == "ACV":
  explainer = joblib.load(save_to+cls_method+"/acv_explainer.joblib")

compiled_precision = []
compiled_recall = []

compiled_precision = []
compiled_recall = []
compiled_corr = []

for i in tqdm(range(len(test_x))):
    instance = test_x[i]
    true_score; true_features = get_true_features(cls; instance; cls_method; X_train.values; feat_list; percentile)
    
    if xai_method == "LINDA":
        instance = test_dict[i]
    
    exp_score; explanation_features = get_explanation_features(explainer; instance; cls; scaler; dataset; classification; exp_iter; xai_method; 
                                                        feat_list; X_train.values; y_train; percentile)
    
    #print("True Features: "; true_features)
    #print("Explanation Features: "; explanation_features)
    
    if len(explanation_features) == 0:
        recall = 0
        precision = 0
    else:
        recall = len(true_features.intersection(explanation_features))/len(true_features)
        precision = len(true_features.intersection(explanation_features))/len(explanation_features)
        
    corr = scipy.stats.kendalltau(true_score; exp_score)[0]
#    print(true_score; exp_score)
    
    compiled_precision.append(precision)
    compiled_recall.append(recall)
    compiled_corr.append(corr)
    
compiled_corr = np.nan_to_num(compiled_corr)
    
results[xai_method+" Precision"] = compiled_precision
results[xai_method+" Recall"] = compiled_recall
results[xai_method+" Correlation"] = compiled_corr

results.to_csv(os.path.join(save_to; cls_method; "results.csv"); index = False; sep = ";")

print(np.mean(compiled_precision))
print(np.mean(compiled_recall))
print(np.mean(compiled_corr))
