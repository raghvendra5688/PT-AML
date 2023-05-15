# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import re 

import xgboost as xgb
from sklearn import ensemble
from sklearn import dummy
from sklearn import linear_model
from sklearn import svm
from sklearn import neural_network
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate, KFold
from sklearn.metrics import make_scorer
from sklearn.utils.fixes import loguniform
import scipy
import argparse

from misc import save_model, load_model, regression_results, grid_search_cv
plt.rcParams["font.family"] = "Arial"


# -

def calculate_regression_metrics(labels, predictions):
    return round(metrics.mean_absolute_error(labels, predictions),3),\
            round(metrics.mean_squared_error(labels, predictions, squared=False),3),\
            round(np.power(scipy.stats.pearsonr(np.array(labels).flatten(),np.array(predictions.flatten()))[0],2),3),\
            round(scipy.stats.pearsonr(np.array(labels).flatten(),np.array(predictions.flatten()))[0],3),\
            round(scipy.stats.spearmanr(np.array(labels).flatten(),np.array(predictions.flatten()))[0],3)


def get_CV_results (model, X_train, Y_train, n_splits):
    kf = KFold(n_splits=n_splits)
    mae_list, rmse_list, r2_list, pr_list, sr_list = [],[],[],[],[]
    for train_index, test_index in kf.split(Y_train):
        y_pred = model.best_estimator_.predict(X_train.iloc[test_index,:])
        results=calculate_regression_metrics(Y_train[test_index],y_pred)
        mae_list.append(results[0])
        rmse_list.append(results[1])
        r2_list.append(results[2])
        pr_list.append(results[3])
        sr_list.append(results[4])
    print(mae_list,rmse_list,r2_list)
    mean_mae, sd_mae = round(np.mean(mae_list),3), round(np.std(mae_list),3)
    mean_rmse, sd_rmse = round(np.mean(rmse_list),3), round(np.std(rmse_list),3)
    mean_r2, sd_r2 = round(np.mean(r2_list),3), round(np.std(r2_list),3)
    mean_pr, sd_pr = round(np.mean(pr_list),3), round(np.std(pr_list),3)
    mean_sr, sd_sr = round(np.mean(sr_list),3), round(np.std(sr_list),3)
    return(mean_mae, sd_mae, mean_rmse, sd_rmse, mean_r2, sd_r2, mean_pr, sd_pr, mean_sr, sd_sr)


#Get the setting with different X_trains and X_tests
train_options = ["../Data/dose_response_with_full_data_inflammasome_with_ls_train.pkl",
                 "../Data/dose_response_with_full_data_inflammasome_with_mfp_train.pkl",
                 ".."]
test_options = ["../Data/dose_response_with_full_data_inflammasome_with_ls_test.pkl",
                "../Data/dose_response_with_full_data_inflammasome_with_mfp_test.pkl",
                ".."]
data_type_options = ["LS_LS","MFP_LS"]
data_information = [('MFP',list(range(510,766))), 
                    ('Pathway', list(range(0,7))+list(range(503,510))),
                    ('Genomics', list(range(7,503))),
                    ('Pathway_Genomics',list(range(0,510))),
                    ('MFP_Pathway', list(range(0,7))+list(range(503,510))+list(range(510,766))),
                    ('MFP_Genomics', list(range(7,503))+list(range(510,766)))]

# +
#Choose the options
input_option1, input_option2 = 1, 0                                                  #Choose 0 for LS for Drug and LS for Cell Line , 1 for MFP for Drug and LS for Cell Line 
classification_task = False
data_type = data_type_options[input_option1]
data_info = data_information[input_option2]

#Get the data for your choice: LS or MFP
print("Loaded training file")
big_train_df = pd.read_pickle(train_options[input_option1],compression="zip")
big_test_df = pd.read_pickle(test_options[input_option1],compression="zip")
total_length = len(big_train_df.columns)
metadata_X_train, X_train, Y_train = big_train_df.loc[:,['ARXSPAN_ID','DRUG_NAME']], big_train_df.iloc[:, range(16,total_length)], big_train_df["y_ic50"].to_numpy().flatten()
metadata_X_test, X_test, Y_test = big_test_df.loc[:,['ARXSPAN_ID','DRUG_NAME']], big_test_df.iloc[:,range(16,total_length)], big_test_df["y_ic50"].to_numpy().flatten()
# -

#LR model
lr_gs = load_model("../Models/lr_models/lr_"+data_type+"_regressor_gs.pk")
scaler = load_model("../Models/lr_models/lr_"+data_type+"_scaling_gs.pk")
X_train_copy = pd.DataFrame(scaler.transform(X_train))
lr_out = get_CV_results(lr_gs, X_train_copy, Y_train, n_splits=5)
print(lr_out)

#GLR Model (MFP + LS), (LS + LS)
glr_gs = load_model("../Models/glr_models/glr_"+data_type+"_regressor_gs.pk")
scaler = load_model("../Models/glr_models/glr_"+data_type+"_scaling_gs.pk")
X_train_copy = pd.DataFrame(scaler.transform(X_train))
glr_out = get_CV_results(glr_gs, X_train_copy, Y_train, n_splits=5)
print(glr_out)

#SVM Model
svr_gs = load_model("../Models/svr_models/svr_"+data_type+"_regressor_gs.pk")
scaler = load_model("../Models/svr_models/svr_"+data_type+"_scaling_gs.pk")
X_train_copy = pd.DataFrame(scaler.transform(X_train))
svr_out = get_CV_results(svr_gs, X_train_copy, Y_train, n_splits=5)
print(svr_out)

#RF Model
rf_gs = load_model("../Models/rf_models/rf_"+data_type+"_regressor_gs.pk")
rf_out = get_CV_results(rf_gs, X_train, Y_train, n_splits=5)
print(rf_out)

#DNN Model
nn_gs = load_model("nn_models/nn_"+data_type+"_regressor_gs.pk")
scaler = load_model("nn_models/nn_"+data_type+"_scaling_gs.pk")
X_train_copy = pd.DataFrame(scaler.transform(X_train))
nn_out = get_CV_results(nn_gs, X_train_copy, Y_train, n_splits=5)
print(nn_out)

# +
#For xgboost model we need to remove brackets from column names
model="xgboost"
if (model=="xgboost" or model=="lgbm"):
    X_train.columns = X_train.columns.str.replace('[','')
    X_train.columns = X_train.columns.str.replace(']','')
    X_test.columns = X_test.columns.str.replace('[','')
    X_test.columns = X_test.columns.str.replace(']','')
    
#XGB Model
xgb_gs = load_model("../Models/xgb_models/xgb_"+data_type+"_regressor_gs.pk")
xgb_out = get_CV_results(xgb_gs, X_train, Y_train, n_splits=5)
print(xgb_out)
# -

#LGBM Model
lgbm_gs = load_model("../Models/lgbm_models/lgbm_"+data_type+"_regressor_gs.pk")
lgbm_out = get_CV_results(lgbm_gs, X_train, Y_train, n_splits=5)
print(lgbm_out)

#Perform CV for XGB ablation study
rev_X_train = X_train.iloc[:,data_info[1]]
rev_X_test = X_test.iloc[:,data_info[1]]
ab_xgb_gs = load_model("../Models/xgb_models/xgb_"+data_info[0]+"_regressor_gs.pk")
ab_xgb_out = get_CV_results(ab_xgb_gs, rev_X_train, Y_train, n_splits=5)
print(ab_xgb_out)


