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

import numpy as np
import pandas as pd
import random
import time
from sklearn.utils import shuffle
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.utils.data as data_utils
import random
from torch.utils.data import DataLoader
from torch.nn.functional import relu,leaky_relu
from torch.nn import Linear
from torch.nn import BatchNorm1d
import networkx as nx
from rdkit import Chem
from torch_geometric import data as DATA
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from math import sqrt
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import pickle
import argparse
from dl_model_architecture import NN_Encoder, GATNet, CNN_Encoder, LSTM_Encoder, Seq2Func_Net, init_weights, count_parameters, training, training_net, evaluation, evaluation_net, epoch_time

from misc import save_model, load_model, regression_results, grid_search_cv, calculate_regression_metrics, supervised_learning_steps, get_CV_results


# +
#Convert SMILES to graph representation
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if(mol is None):
        return None
    else:
        c_size = mol.GetNumAtoms()
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append( feature / sum(feature) )

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
        
        return c_size, features, edge_index

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def get_smiles_func(smiles, cell_features, labels):
    #Create smiles graphs
    data_list=[]
    for i in range(0,len(smiles)):
        smile = smiles[i]
        label = labels[i]
        cell_feature = cell_features[i]
        g = smile_to_graph(smile)
        if(g is None):
            print(smile)
            none_smiles.append(smile)
        else:
            c_size, features, edge_index = g[0],g[1],g[2]
            
            GCNData = DATA.Data(x=torch.FloatTensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([label]))
            GCNData.cell_src = torch.FloatTensor(cell_feature)
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)
    return(data_list)


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
#Convert the cell line info into scaled vectors
scaler = preprocessing.StandardScaler()
X_train_copy = scaler.fit_transform(rev_X_train)
X_test_copy = scaler.transform(rev_X_test)

#Get the list of training smiles
X_train_smiles = X_train["CanonicalSMILES"].tolist()
X_train_smiles_graphs = get_smiles_func(X_train_smiles,X_train_copy,Y_train)
X_test_smiles = X_test["CanonicalSMILES"].tolist()
X_test_smiles_graphs = get_smiles_func(X_test_smiles,X_test_copy,Y_test)
# -
N_dim = rev_X_train.shape[1]
for i in range(0,10):
    #Split the data into 0.8 for training and rest for validation stuff
    BATCH_SIZE = 256
    train_dataset, valid_dataset = data_utils.random_split(X_train_smiles_graphs, [0.8, 0.2], generator = torch.Generator().manual_seed(i*42))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(X_test_smiles_graphs, batch_size=BATCH_SIZE, shuffle=False)
            
    #Build model parameters
    CELL_INPUT_DIM = X_train_copy.shape[1]
    CELL_OUT_DIM = 256
    CELL_HID_DIMS = [1024,512]

    SMILES_INPUT_DIM = 78
    SMILES_N_HEAD = 2
    SMILES_HID_DIM = 256
    SMILES_OUT_DIM = 256

    HID_DIM = 128
    OUT_DIM = 1
    DROPOUT = 0.2

    cell_enc = NN_Encoder(CELL_INPUT_DIM, CELL_OUT_DIM, CELL_HID_DIMS, DROPOUT)
    smiles_enc = GATNet(SMILES_INPUT_DIM, SMILES_N_HEAD, SMILES_HID_DIM, SMILES_OUT_DIM, DROPOUT)

    #Make the model
    model = Seq2Func_Net(cell_enc, smiles_enc, HID_DIM, OUT_DIM, DROPOUT, device=DEVICE).to(DEVICE)
    print("Total parameters in model are: ",count_parameters(model))
    model.apply(init_weights)
    #print(model)
    
    #Model training criterion
    optimizer = optim.Adam(model.parameters(),weight_decay=1e-4)
    criterion = nn.MSELoss().to(DEVICE)
    
    #Start training the model
    outputfile_model = '../Models/gat_models/gat_supervised_checkpoint_'+str(i)+'.pt'
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
            train_loss = training_net(model, train_loader, optimizer, criterion, N_dim, CLIP, DEVICE)
            valid_loss = evaluation_net(model, valid_loader, criterion, N_dim, DEVICE)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            if valid_loss < best_valid_loss:
                counter = 0
                print("Current Val. Loss: %.3f better than prev Val. Loss: %.3f " %(valid_loss,best_valid_loss))
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), outputfile_model)
        else:
            counter+=1
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')

    if (torch.cuda.is_available()):
        model.load_state_dict(torch.load(outputfile_model))
    else:
        model.load_state_dict(torch.load(outputfile_model,map_location=torch.device('cpu')))
        
    valid_loss = evaluation_net(model, valid_loader, criterion, N_dim, DEVICE)
    print(f'| Best Valid Loss: {valid_loss:.3f}')

    test_loss = evaluation_net(model, test_loader, criterion, N_dim, DEVICE)
    print(f'| Test Loss: {test_loss: .3f}')

    fout_filename = "../Models/gat_models/gat_supervised_"+str(i)+"_loss_plot.csv"
    fout=open(fout_filename,"w")
    for j in range(len(train_loss_list)):
        outputstring = str(train_loss_list[j])+","+str(valid_loss_list[j])+"\n"
        fout.write(outputstring)
    fout.close()


