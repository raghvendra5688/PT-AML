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
from dl_model_architecture import NN_Encoder, CNN_Encoder, Seq2Func, init_weights, count_parameters, training, evaluation, epoch_time

from misc import save_model, load_model, regression_results, grid_search_cv, calculate_regression_metrics, supervised_learning_steps, get_CV_results
# -

#Setting up the environment
SEED = 123
random.seed(SEED)
st = random.getstate()
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.is_available()
cudaid = int(0)
DEVICE = torch.device("cuda:%d" % (cudaid) if torch.cuda.is_available() else "cpu")
print(DEVICE)

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
tokenizer = SmilesTokenizer("/home/brc05/TII/Projects/Immunoinformatics/Data/vocab.txt")

max_smiles_length=150
def encode_to_indices(x):
    return(torch.tensor(tokenizer.encode(x,max_length=max_smiles_length,padding="max_length")))

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
BATCH_SIZE = 4096
train_dataset, valid_dataset = data_utils.random_split(train, [0.8, 0.2], generator = torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

# +
#Build model parameters
CELL_INPUT_DIM = X_train_copy.shape[1]
CELL_OUT_DIM = 256
CELL_HID_DIMS = [1024,512]

SMILES_INPUT_DIM = tokenizer.vocab_size
SMILES_ENC_EMB_DIM = 128
SMILES_OUT_DIM = 256

N_FILTERS = 64
FILTER_SIZES = [2,3,4,6,7,8,9,10]

HID_DIM = 256
OUT_DIM = 1
DROPOUT = 0.2

cell_enc = NN_Encoder(CELL_INPUT_DIM, CELL_OUT_DIM, CELL_HID_DIMS, DROPOUT)
smiles_enc = CNN_Encoder(SMILES_INPUT_DIM, SMILES_ENC_EMB_DIM, SMILES_OUT_DIM, N_FILTERS, FILTER_SIZES, DROPOUT)

#Make the model
model = Seq2Func(cell_enc, smiles_enc, HID_DIM, OUT_DIM, DROPOUT, device=DEVICE).to(DEVICE)
print("Total parameters in model are: ",count_parameters(model))
model.apply(init_weights)
# -

#Model training criterion
optimizer = optim.Adam(model.parameters(),weight_decay=1e-4)
criterion = nn.MSELoss().to(DEVICE)

# +
#Start training the model
N_EPOCHS = 5000
CLIP = 1
counter = 0
patience = 1000
train_loss_list = []
valid_loss_list = []
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    if (counter<patience):
        print("Counter Id: ",str(counter))
        start_time = time.time()
        train_loss = training(model, train_loader, optimizer, criterion, CLIP, DEVICE)
        valid_loss = evaluation(model, valid_loader, criterion, DEVICE)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        if valid_loss < best_valid_loss:
            counter = 0
            print("Current Val. Loss: %.3f better than prev Val. Loss: %.3f " %(valid_loss,best_valid_loss))
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), '../Models/cnn_models/cnn_supervised_checkpoint.pt')
        else:
            counter+=1
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')

if (torch.cuda.is_available()):
    model.load_state_dict(torch.load('../Models/cnn_models/cnn_supervised_checkpoint.pt'))
else:
    model.load_state_dict(torch.load('../Models/cnn_models/cnn_supervised_checkpoint.pt',map_location=torch.device('cpu')))
        
valid_loss = evaluation(model, valid_loader, criterion, DEVICE)
print(f'| Best Valid Loss: {valid_loss:.3f}')

test_loss = evaluation(model, test_loader, criterion, DEVICE)
print(f'| Test Loss: {test_loss: .3f}')

fout=open("../Models/cnn_model/cnn_supervised_loss_plot.csv","w")
for i in range(len(train_loss_list)):
    outputstring = str(train_loss_list[i])+","+str(valid_loss_list[i])+"\n"
    fout.write(outputstring)
    fout.close()
# -


