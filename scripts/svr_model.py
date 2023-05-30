# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
from sklearn import kernel_approximation
from sklearn.kernel_approximation import Nystroem
from sklearn import neural_network
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import loguniform
import scipy
import argparse

from misc import save_model, load_model, regression_results, grid_search_cv, supervised_learning_steps, regression_results, calculate_regression_metrics, get_CV_results
# -

#Get the setting with different X_trains and X_tests
train_options = ["../Data/Training_Set_with_Drug_Embedding_Cell_Info.pkl",
                 "../Data/Training_Set_with_Drug_MFP_Cell_Info.pkl",
                 ".."]
test_options = ["../Data/Test_Set_with_Drug_Embedding_Cell_Info.pkl",
                "../Data/Test_Set_with_Drug_MFP_Cell_Info.pkl",
                ".."]
data_type_options = ["LS_Feat","MFP_Feat"]

# +
#Choose the options
input_option = 1                                                  #Choose 0 for LS for Drug and LS for Cell Line , 1 for MFP for Drug and LS for Cell Line 
classification_task = False
data_type = data_type_options[input_option]

#Get the data for your choice: LS or MFP
print("Loaded training file")
big_train_df = pd.read_pickle(train_options[input_option],compression="zip")
big_test_df = pd.read_pickle(test_options[input_option],compression="zip")
total_length = len(big_train_df.columns)
if (input_option==0):
    #Consider only those columns which have numeric values
    metadata_X_train,X_train, Y_train = big_train_df.loc[:,["dbgap_rnaseq_sample","inhibitor"]], big_train_df.iloc[:,[1,4]+[*range(6,262,1)]+[*range(288,total_length,1)]], big_train_df["auc"].to_numpy().flatten()
    metadata_X_test,X_test, Y_test = big_test_df.loc[:,["dbgap_rnaseq_sample","inhibitor"]], big_test_df.iloc[:,[1,4]+[*range(6,262,1)]+[*range(288,total_length,1)]], big_test_df["auc"].to_numpy().flatten()
elif (input_option==1):
    metadata_X_train,X_train, Y_train = big_train_df.loc[:,["dbgap_rnaseq_sample","inhibitor"]], big_train_df.iloc[:,[1,4]+[*range(6,1030,1)]+[*range(1056,total_length,1)]], big_train_df["auc"].to_numpy().flatten()
    metadata_X_test,X_test, Y_test = big_test_df.loc[:,["dbgap_rnaseq_sample","inhibitor"]], big_test_df.iloc[:,[1,4]+[*range(6,1030,1)]+[*range(1056,total_length,1)]], big_test_df["auc"].to_numpy().flatten()

#Keep only numeric training and test set and those which have no Nans
X_train_numerics_only = X_train.select_dtypes(include=np.number)
X_test_numerics_only = X_test[X_train_numerics_only.columns]
print("Shape of training set after removing non-numeric cols")
print(X_train_numerics_only.shape)
print(X_test_numerics_only.shape)


nan_cols = [i for i in X_train_numerics_only.columns if X_train_numerics_only[i].isnull().any()]
rev_X_train = X_train_numerics_only.drop(nan_cols,axis=1)
rev_X_test = X_test_numerics_only.drop(nan_cols,axis=1)
print("Shape of training set after removing cols with NaNs")
print(rev_X_train.shape)
print(rev_X_test.shape)

# +
#Build the SVR
model = svm.LinearSVR(random_state=42, dual=False, loss='squared_epsilon_insensitive')

# Grid parameters
params_svr = [
    {
        'C': loguniform(1e-2, 1e1),
        'epsilon': scipy.stats.uniform.rvs(size=20, random_state=42),
        'max_iter': [5000,10000]
    }
]
#It will select 100 random combinations for the CV and do 5-fold CV for each combination
n_iter = 100
scaler = preprocessing.StandardScaler()
X_train_copy = scaler.fit_transform(rev_X_train)

#Build the linear SVM model after doing Nystrom approx.
svr_gs=supervised_learning_steps("svr","r2",data_type,classification_task,model,params_svr,X_train_copy,Y_train,n_iter=n_iter,n_splits=5)
        
#Build the model and get 5-fold CV results    
#print(svr_gs.cv_results_)
save_model(scaler, "%s_models/%s_%s_scaling_gs.pk" % ("svr","svr",data_type))
# -

svr_gs = load_model("svr_models/svr_"+data_type+"_regressor_gs.pk")
scaler = load_model("svr_models/svr_"+data_type+"_scaling_gs.pk")
X_train_copy = scaler.transform(rev_X_train)
results=get_CV_results(svr_gs, pd.DataFrame(X_train_copy), Y_train, n_splits=5)
print(results)

# +
#Test the svm model on separate test set 
svr_gs = load_model("svr_models/svr_"+data_type+"_regressor_gs.pk")
scaler = load_model("svr_models/svr_"+data_type+"_scaling_gs.pk")
np.max(svr_gs.cv_results_["mean_test_score"])
svr_best = svr_gs.best_estimator_
y_pred_svr=svr_best.predict(scaler.transform(rev_X_test))
test_metrics=calculate_regression_metrics(Y_test,y_pred_svr)
print(test_metrics)

#Write the prediction of SVR model
metadata_X_test['predictions']=y_pred_svr
metadata_X_test['labels']=Y_test
metadata_X_test.to_csv("../Results/SVR_"+data_type+"_supervised_test_predictions.csv",index=False)
print("Finished writing predictions")

fig = plt.figure()
plt.style.use('classic')
fig.set_size_inches(2.5,2.5)
fig.set_dpi(300)
fig.set_facecolor("white")

ax = sn.regplot(x="labels", y="predictions", data=metadata_X_test, scatter_kws={"color": "lightblue",'alpha':0.5}, 
                line_kws={"color": "red"})
ax.axes.set_title("SVR Predictions (MFP + Feat)",fontsize=10)
ax.set_xlim(0, 300)
ax.set_ylim(0, 300)
ax.set_xlabel("Label",fontsize=10)
ax.set_ylabel("Prediction",fontsize=10)
ax.tick_params(labelsize=10, color="black")
plt.text(25, 25, 'Pearson r =' +str(test_metrics[3]), fontsize = 10)
plt.text(25, 50, 'MAE ='+str(test_metrics[0]),fontsize=10)
outfilename = "../Results/SVR_"+data_type+"_supervised_test_prediction.pdf"
plt.savefig(outfilename, bbox_inches="tight")

# +
#Get the top coefficients and matching column information
svr_best = load_model("svr_models/svr_"+data_type+"_regressor_best_estimator.pk")
val, index = np.sort(np.abs(svr_best.coef_)), np.argsort(np.abs(svr_best.coef_))

fig = plt.figure()
plt.style.use('classic')
fig.set_size_inches(4,3)
fig.set_dpi(300)
fig.set_facecolor("white")

ax = fig.add_subplot(111)
plt.bar(rev_X_train.columns[index[-20:]],val[-20:])
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees

ax.axes.set_title("Top SVR Coefficients (MFP + Feat)",fontsize=9)
ax.set_xlabel("Features",fontsize=9)
ax.set_ylabel("Coefficient Values",fontsize=9)
ax.tick_params(labelsize=9)
outputfile = "../Results/SVR_"+data_type+"_Coefficients.pdf"
plt.savefig(outputfile, bbox_inches="tight")
# -


