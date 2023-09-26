import pandas as pd
import numpy as np

from DatasetManager import DatasetManager

from sklearn.metrics import f1_score, classification_report, roc_auc_score, r2_score, mean_absolute_percentage_error
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
import multiprocessing as mp

import json
from collections import Counter

def get_reg_features(cls, percentile):
    
    og_coef = cls.coef_
    if len(og_coef.shape) > 1:
        og_coef = og_coef[0]
    
    coef = [abs(val) for val in og_coef]
    
    min_coef = min(coef)
    max_coef = max(coef)
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k
    
    feat_pos = [i for i in range(len(coef)) if coef[i] >= q1_min]
    
    return coef, feat_pos

def get_nb_features(cls, instance, percentile):
    pred = cls.predict(instance.reshape(1, -1))
    means = cls.theta_[pred][0]
    std = np.sqrt(cls.var_[pred])[0]

    alt = 1-pred
    alt_means = cls.theta_[alt][0]
    alt_std = np.sqrt(cls.var_[alt])[0]
    
    likelihoods = []
    
#     for i in range(len(means)):
#         lkhood = scipy.stats.norm(means[i], std[i]).logpdf(instance[i])
#         #likelihoods.append(abs(lkhood))
#         likelihoods.append(lkhood)

    for i in range(len(means)):
        lk = scipy.stats.norm(means[i], std[i]).logpdf(instance[i])
        alt_lk = scipy.stats.norm(alt_means[i], alt_std[i]).logpdf(instance[i])
        lkhood = abs(lk-alt_lk)
        likelihoods.append(lkhood)
    
    min_coef = min(likelihoods)
    max_coef = max(likelihoods)
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k
    
#     bins = pd.cut(likelihoods, 10, retbins = True, duplicates = "drop")[1]
#     lim_1 = bins[-2]
#     lim_2 = bins[1]
    
#     sortedls = sorted(likelihoods, reverse=True)
#     pos = math.ceil(len(likelihoods)/4)
#     lim = likelihoods[pos]
    
    feat_pos = [i for i in range(len(likelihoods)) if likelihoods[i] >= q1_min]# or likelihoods[i] <= lim_2]
    
    return likelihoods, feat_pos

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
            
            
    feat_pos = set(feats)
    
    score = np.zeros(len(instance))
    n = len(feats)
    for i in feats:
        score[i]+=n
        n=n-1
    
    return score, feat_pos

def get_path_depths(tree, feat_list, cur_depth = 0, lvl = 0, depths = []):
    
    left_child = tree.children_left[lvl]
    right_child = tree.children_right[lvl]
    
    if left_child == sklearn.tree._tree.TREE_LEAF:
        depths.append(cur_depth)
        
    else:
        depths = get_path_depths(tree, feat_list, cur_depth+1, left_child, depths)
        depths = get_path_depths(tree, feat_list, cur_depth+1, right_child, depths)
    return depths

def get_shap_features(explainer, instance, cls, classification, exp_iter, feat_list, percentile):
    
    shap_exp = []
    
    pred = cls.predict(instance.reshape(1, -1))
    
    for i in range(exp_iter):
        if type(explainer) == shap.explainers._tree.Tree:
            exp = explainer(instance, check_additivity = False).values
        else:
            exp = explainer(instance.reshape(1, -1)).values
        
        #print(exp.shape)
        #print(exp)
        
        if exp.shape == (1, len(feat_list), 2):
            exp = exp[0]
            
        #print(exp.shape)
        
        if exp.shape == (len(feat_list), 2):
            exp = np.array([feat[pred] for feat in exp]).reshape(len(feat_list))
        elif exp.shape == (1, len(feat_list)) or exp.shape == (len(feat_list), 1):
            exp = exp.reshape(len(feat_list))
            
        #print(np.array(exp).shape)
            
        shap_exp.append(exp)
        
    #print(np.array(shap_exp).shape)
        
    if np.array(shap_exp).shape != (exp_iter, len(feat_list)):
        raise Exception("Explanation shape is not correct. It is", np.array(shap_exp).shape, "instead of the expected", (exp_iter, len(feat_list)))
    
#     if classification==True, type(explainer) == shap.explainers._tree.Tree:
#         shap_exp = []
#         for each in full_exp:
#             single_exp = [feat[0] for feat in each]
#             shap_exp.append(single_exp)
#     else:
#         shap_exp = full_exp
        
    avg_val = np.average(shap_exp, axis = 0)
    abs_val = [abs(val) for val in avg_val]
    
    #Get recall and precision for the average of shap values
    min_coef = min(abs_val)
    max_coef = max(abs_val)
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k

    sorted_val = np.copy(abs_val)
    sorted_val.sort()
    
    shap_features = set([i for i in range(len(feat_list)) if abs_val[i] > q1_min])
    
    return abs_val, shap_features

def get_lime_features(explainer, instance, cls, classification, exp_iter, feat_list, percentile):
    lime_exp = []
    
    for i in range(exp_iter):
        if classification==True:
            lime_exp.extend(explainer.explain_instance(instance, cls.predict_proba, 
                                                num_features=len(feat_list), labels=[0,1]).as_list())
        else:
            lime_exp.extend(explainer.explain_instance(instance, cls.predict, 
                                                num_features=len(feat_list), labels=[0,1]).as_list())
            
    weights = [[] for each in feat_list]
    for exp in lime_exp:
        feat = exp[0]
        if '<' in feat:
            feat = exp[0].replace("= ",'')
            parts = feat.split('<')
        elif '>' in feat:
            feat = exp[0].replace("= ",'')
            parts = feat.split('>')
        else:
            parts = feat.split("=")
        
        for part in parts:
            if part.replace('.','').replace(' ','').lstrip('-').isdigit()==False:
                feat_name = part.replace(' ','')
        n = feat_list.index(feat_name)
        weights[n].append(exp[1])
    
    weights = np.transpose(weights)
    avg_weight = np.average(np.array(weights), axis = 0)
    abs_weight = [abs(weight) for weight in avg_weight]
    
    min_coef = min(abs_weight)
    max_coef = max(abs_weight)
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k
    
    sorted_weight = np.copy(abs_weight)
    sorted_weight.sort()
    
    lime_features = set([i for i in range(len(feat_list)) if abs_weight[i] >= q1_min])
    
    return abs_weight, lime_features

def get_linda_features(instance, cls, scaler, dataset, exp_iter, feat_list, percentile):
    label_lst = ["Negative", "Positive"]
    
    feat_pos = []
    lkhoods = []
    
    save_to = os.path.join(PATH, dataset, cls_method, method_name)+"/"
    
    for i in range(exp_iter):
        [bn, inference, infoBN] = generate_BN_explanations(instance, label_lst, feat_list, "Result", 
                                                           None, scaler, cls, save_to, dataset, show_in_notebook = False,
                                                           samples=round(len(feat_list)*2))
        
        ie = pyAgrum.LazyPropagation(bn)
        result_posterior = ie.posterior(bn.idFromName("Result")).topandas()
        if len(result_posterior.shape)==1:
            result_proba = result_posterior.values[0]
        else:
            result_proba = result_posterior.loc["Result", label_lst[instance['predictions']]]        
        row = instance['original_vector']
        #print(row)

        likelihood = [0]*len(feat_list)

        for j in range(len(feat_list)):
            var_labels = bn.variable(feat_list[j]).labels()
            str_bins = list(var_labels)
            bins = []

            for disc_bin in str_bins:
                disc_bin = disc_bin.strip('"(]')
                cat = [float(val) for val in disc_bin.split(',')]
                bins.append(cat)

            feat_bin = None
            val = row[j]
            
            #Find appropriate bin, if higher or lower than bins,
            #use first or last bin
            for k in range(len(bins)):
                if k == 0 and val <= bins[k][0]:
                    feat_bin = str_bins[k]
                elif k == len(bins)-1 and val >= bins[k][1]:
                    feat_bin = str_bins[k]
                elif val > bins[k][0] and val <= bins[k][1]:
                    feat_bin = str_bins[k]

            #If the value doesn't fit into any bin,
            #pick the nearest
            if feat_bin == None: 
                bins_diff = np.array(bins) - val
                inds = np.unravel_index(np.abs(bins_diff).argmin(axis=None), bins_diff.shape)
                k = inds[0]
                feat_bin = str_bins[k]
            
            result_posterior = ie.posterior(bn.idFromName("Result")).topandas()
            if len(result_posterior.shape)==1:
                new_proba = result_posterior.values[0]
            else:
                new_proba = result_posterior.loc["Result", label_lst[instance['predictions']]]
            #print(result_proba, new_proba)
            proba_change = result_proba-new_proba
            likelihood[j] = abs(proba_change)

        lkhoods.append(likelihood)
        
    min_coef = min( np.mean(lkhoods, axis=0))
    max_coef = max( np.mean(lkhoods, axis=0))
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k

    #If fixing all features produces the same result for the class,
    #return all features
    if len(set(np.mean(lkhoods, axis=0)))==1:
        feat_pos.extend(range(len(feat_list)))
    else:
        feat_pos.extend(list(np.where(np.mean(lkhoods, axis=0) >= q1_min)[0]))

    feat_pos = set(feat_pos)
        
    return np.mean(lkhoods, axis=0), feat_pos

def get_acv_features(explainer, instance, cls, X_train, y_train, exp_iter):
    instance = instance.reshape(1, -1)
    y = cls.predict(instance)
    
    t=np.var(y_train)

    feats = []
    feat_imp = []

    for i in range(exp_iter):
        sufficient_expl, sdp_expl, sdp_global = explainer.sufficient_expl_rf(instance, y, X_train, y_train,
                                                                                 t=t, pi_level=0.8)
        clean_expl = sufficient_expl.copy()
        clean_expl = clean_expl[0]
        clean_expl = [sublist for sublist in clean_expl if sum(n<0 for n in sublist)==0 ]

        clean_sdp = sdp_expl[0].copy()
        clean_sdp = [sdp for sdp in clean_sdp if sdp > 0]
        
        lximp = explainer.compute_local_sdp(X_train.shape[1], clean_expl)
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
    
      
    feat_imp = np.mean(feat_imp, axis=0)
    
    return feat_imp, feat_pos

def get_explanation_features(explainer, instance, cls, scaler, dataset, 
                             classification, exp_iter, xai_method, feat_list, X_train, y_train, percentile):
    if xai_method == "SHAP":
        exp_score, feat_pos = get_shap_features(explainer, instance, cls, classification, exp_iter, feat_list, percentile)
        
    elif xai_method == "LIME":
        exp_score, feat_pos = get_lime_features(explainer, instance, cls, classification, exp_iter, feat_list, percentile)
        
    elif xai_method == "LINDA":
        exp_score, feat_pos = get_linda_features(instance, cls, scaler, dataset, exp_iter, feat_list, percentile)

    elif xai_method == "ACV":
        exp_score, feat_pos = get_acv_features(explainer, instance, cls, X_train, y_train, exp_iter)
        
    explanation_features = [feat_list[i] for i in feat_pos]
    #explanation_features = set(explanation_features)
        
    return exp_score, explanation_features

def get_true_features(cls, instance, cls_method, X_train, feat_list, percentile):
    if cls_method == "decision_tree":
        true_score, feat_pos = get_tree_features(cls, instance)
        
    elif cls_method == "logit" or cls_method == "lin_reg":
        true_score, feat_pos = get_reg_features(cls, percentile)
        
    elif cls_method == "nb":
        true_score, feat_pos = get_nb_features(cls, instance, percentile)
        
    true_features = [feat_list[i] for i in feat_pos]
    true_features = set(true_features)
    
    #print(feat_pos)
    
    return true_score, true_features

def test_fidelity(i):
    #print("i:", i)
    instance = sample_instances[i]
    #print(instance)
    true_score, true_features = get_true_features(cls, instance, cls_method, trainingdata, feat_list, percentile)
    #print("True Score", true_score)
    
    if xai_method == "LINDA":
        instance = test_dict[i]
    
    exp_score, explanation_features = get_explanation_features(explainer, instance, cls, scaler, dataset_ref, classification, exp_iter, xai_method, 
                                                        feat_list, trainingdata, targets, percentile)
#     print("Exp Score", exp_score)
#     print("True Features: ", true_features)
#     print("Explanation Features: ", explanation_features)
    
    if len(explanation_features) == 0:
        recall = 0
        precision = 0
    else:
        recall = len(true_features.intersection(explanation_features))/len(true_features)
        precision = len(true_features.intersection(explanation_features))/len(explanation_features)
        
#     print("Recall:", recall)
#     print("Precision:", precision)
        
    corr = scipy.stats.kendalltau(true_score, exp_score)[0]
    
#     print("Corr:", corr)
    #progress_bar.clear()
    #progress_bar.update(i)
    
    return precision, recall, corr

# path to project folder
# please change to your own
PATH = os.getcwd()
sys.path.append(PATH)

dataset_ref = sys.argv[1]
params_dir = PATH + "params"
results_dir = "results"
bucket_method = sys.argv[2]
cls_encoding = sys.argv[3]
cls_method = sys.argv[4]

classification = True

gap = 1
n_iter = 1

method_name = "%s_%s"%(bucket_method, cls_encoding)
save_to = os.path.join(PATH, dataset_ref, cls_method, method_name)

xai_method = sys.argv[5]

sample_size = 2
exp_iter = 5
max_feat = 10
max_prefix = 20
random_state = 22
percentile = 0.05

dataset_ref_to_datasets = {
    "bpic2012" : ["bpic2012_accepted"],
    "sepsis_cases": ["sepsis_cases_1"],
    "production" : ["production"],
    "bpic2011": ["bpic2011_f1"],
    "hospital": ["hospital_billing_2"],
    "traffic": ["traffic_fines_1"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]

all_results = []
for dataset_name in datasets:
        
    for ii in range(n_iter):
        num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s/%s/pipelines'% 
                                                                    (dataset_ref, cls_method, method_name)))])
        dataset_manager = DatasetManager(dataset_name)

        for bucket in tqdm(range(num_buckets)):
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
# 
            #import training data for bucket
            trainingdata = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/train_data_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID))).values
            targets = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/y_train_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID))).values
            if scaler != None:
                trainingdata = scaler.transform(trainingdata)

            #find relevant samples for bucket
            sample_instances = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/test_sample_bucket_%s.csv" % 
                                      (dataset_ref, cls_method, method_name, bucketID))).values
            results = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/results_bucket_%s.csv" % 
                                      (dataset_ref, cls_method, method_name, bucketID)), sep=";")

            if scaler != None:
                sample_instances = scaler.transform(sample_instances)
                
            #Identify feature names
            feat_list = [feat.replace(" ", "_") for feat in feature_combiner.get_feature_names()]
            class_names = ["Negative", "Positive"]
            cats = [feat for col in dataset_manager.dynamic_cat_cols+dataset_manager.static_cat_cols 
                    for feat in range(len(feat_list)) if col in feat_list[feat]]
            
            #create explanation mechanism
            if xai_method == "SHAP":
                if cls_method == "xgboost" or cls_method == "decision_tree":
                    explainer = shap.Explainer(cls)
                elif cls_method == "nb":
                    explainer = shap.Explainer(cls.predict_proba, trainingdata)
                else:
                    explainer = shap.Explainer(cls, trainingdata)
            elif xai_method == "LIME":
                explainer = lime.lime_tabular.LimeTabularExplainer(trainingdata,
                                  feature_names = feat_list, class_names=class_names, categorical_features = cats)
            elif xai_method == "ACV":
                explainer = joblib.load(os.path.join(PATH,'%s/%s/%s/acv_surrogate/acv_explainer_bucket_%s.joblib'% 
                                                                    (dataset_ref, cls_method, method_name, bucketID)))
            elif xai_method == "LINDA":
                test_dict = generate_local_predictions( sample_instances, results["Actual"], cls, scaler, None )
                explainer = None

            compiled_precision = []
            compiled_recall = []
            compiled_corr = []

            #for i in tqdm_notebook(range(len(test_x))):
            #explain the chosen instances and find the fidelity
            pool = mp.Pool(mp.cpu_count(), initargs=(mp.RLock(),), initializer=tqdm.set_lock)

            #with tqdm(total = len(test_x)) as progress_bar:
            start = time.time()
            #print(start)
            #recall, precision, corr = zip(*pool.map(test_fidelity, [i for i in range(len(test_x))]))
            for result in tqdm(pool.imap(test_fidelity, [i for i in range(len(sample_instances))]), total = len(sample_instances)):
                compiled_precision.append(result[0])
                compiled_recall.append(result[1])
                compiled_corr.append(result[2])

            print(time.time()-start, "seconds")

            # compiled_precision = list(precision)
            # compiled_recall = list(recall)
            # compiled_corr = list(corr)

            compiled_corr = np.nan_to_num(compiled_corr)

            results[xai_method+" Precision"] = compiled_precision
            results[xai_method+" Recall"] = compiled_recall
            results[xai_method+" Correlation"] = compiled_corr
            
            print("Average precision:", np.mean(compiled_precision))
            print("Average recall:", np.mean(compiled_recall))
            print("Average correlation:", np.mean(compiled_corr))
            print("\n---------------------\n")
            
            results.to_csv(os.path.join(PATH,"%s/%s/%s/samples/results_bucket_%s.csv") % 
                                (dataset_ref, cls_method, method_name, bucketID), sep=";", index=False) 
            all_results.append(results)
            
pd.concat(all_results).to_csv(os.path.join(PATH,"%s/%s/%s/samples/results.csv") % (dataset_ref, cls_method, method_name), 
                              sep=";", index=False)
