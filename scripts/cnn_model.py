# ---
# jupyter:
#   jupytext:
#     formats: py:light
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
# #!pip install transformers
# #!pip install --pre deepchem
# #!pip install rdkit

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
import lightgbm
import catboost
from sklearn import ensemble
from sklearn import dummy
from sklearn import linear_model
from sklearn import svm
from sklearn import neural_network
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import loguniform
import scipy
import argparse

import random
import math
import time

import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import softmax, relu, selu, elu
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import random
import math
#from seq2func import LSTM_Encoder, CNN_Encoder, Seq2Func, init_weights, count_parameters, train, evaluate, epoch_time

from misc import save_model, load_model, regression_results, grid_search_cv, calculate_regression_metrics, supervised_learning_steps, get_CV_results

# +
#Get the data for your choice: Canonical SMILES + Cell Line Info
print("Loaded training file")
big_train_df = pd.read_pickle("../Data/Training_Set_Var_with_Drug_Embedding_Cell_Info.pkl",compression="zip")
big_test_df = pd.read_pickle("../Data/Test_Set_Var_with_Drug_Embedding_Cell_Info.pkl",compression="zip")
total_length = len(big_train_df.columns)

metadata_X_train,X_train, Y_train = big_train_df.loc[:,["dbgap_rnaseq_sample","inhibitor"]], big_train_df.iloc[:,[2,1,4]+[*range(288,total_length,1)]], big_train_df["auc"].to_numpy().flatten()
metadata_X_test,X_test, Y_test = big_test_df.loc[:,["dbgap_rnaseq_sample","inhibitor"]], big_test_df.iloc[:,[2,1,4]+[*range(288,total_length,1)]], big_test_df["auc"].to_numpy().flatten()

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
#Load the tokenizer and tokenize SMILES using Vocab from DeepChem
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
tokenizer = SmilesTokenizer("/home/raghvendra/TII/Projects/Raghav/Immunoinformatics/Data/vocab.txt")

def encode_to_indices(x):
    return(torch.tensor(tokenizer.encode(x,max_length=150,padding="max_length")))

#Get the list of training smiles
X_train_smiles = X_train["CanonicalSMILES"].tolist()
X_train_smiles_encoded = [encode_to_indices(x) for x in X_train_smiles]
X_test_smiles = X_test["CanonicalSMILES"].tolist()
X_test_smiles_encoded = [encode_to_indices(x) for x in X_test_smiles]

#Convert train and test smiles to stack of tensors
X_train_smiles_encoded = torch.stack(X_train_smiles_encoded)
X_test_smiles_encoded = torch.stack(X_test_smiles_encoded)

#Convert the cell line info into scaled vectors
scaler = preprocessing.StandardScaler()
X_train_copy = scaler.fit_transform(rev_X_train)
X_test_copy = scaler.transform(rev_X_test)

#Create the training and test tensor datasets
train = data_utils.TensorDataset(X_train_smiles_encoded, torch.Tensor(np.array(X_train_copy)),torch.Tensor(np.array(Y_train)))
test = data_utils.TensorDataset(X_test_smiles_encoded, torch.Tensor(np.array(X_test_copy)),torch.Tensor(np.array(Y_test)))
# -
#Split the data into 0.75 for training and rest for validation stuff
train_dataset, valid_dataset = data_utils.random_split(train, [0.75, 0.25], generator = torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test, batch_size=16, shuffle=False)

for i,batch in enumerate(train_loader):
    print("Iteration ",str(i))
    compound_encoding,cell_line_encoding,labels = batch[0],batch[1],batch[2]
    print(compound_encoding.size())
    print(cell_line_encoding.size())
    print(labels.size())
    break


