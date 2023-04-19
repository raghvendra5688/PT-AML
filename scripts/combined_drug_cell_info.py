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

import pandas as pd
import numpy as np
import os
import pickle

#Run the command to generate the train and test pkl files
command = "python preprocess_cellline.py"
os.system(command)

#Load the training and test set with gene expression, clinical traits, pathway activations, celltype and module activations, mutations
train_feature_df = pd.read_pickle("../Data/Training_Set_Mod.pkl",compression="zip")
test_feature_df = pd.read_pickle("../Data/Test_Set_Mod.pkl",compression="zip")
print(train_feature_df.shape)
print(test_feature_df.shape)
print(train_feature_df.columns)

# +
#Load the training and test drug, cell combination file
train_drug_cell_df = pd.read_csv("../Data/Revised_Training_Set_with_IC50.csv.gz",compression="gzip",header='infer',sep="\t")
test_drug_cell_df = pd.read_csv("../Data/Revised_Test_Set_with_IC50.csv.gz",compression="gzip",header="infer",sep="\t")
print(train_drug_cell_df.shape)
print(test_drug_cell_df.shape)

rev_train_feature_df = train_feature_df.iloc[:,[0]+[i for i in range(22844,23322)]]
rev_test_feature_df = test_feature_df.iloc[:,[0]+[i for i in range(22844,23322)]]
print(rev_train_feature_df.columns)
# -

#Merge the dataframes containing drug-cell info and cell line info df
train_drug_cell_feature_df = pd.merge(train_drug_cell_df, rev_train_feature_df, on="dbgap_rnaseq_sample")
print(train_drug_cell_feature_df.shape)
test_drug_cell_feature_df = pd.merge(test_drug_cell_df, rev_test_feature_df, on="dbgap_rnaseq_sample")
print(test_drug_cell_feature_df.shape)

#Write the pickle files
train_drug_cell_feature_df.to_pickle("../Data/Training_Set_with_Drug_Cell_Info.pkl", compression="zip")
test_drug_cell_feature_df.to_pickle("../Data/Test_Set_with_Drug_Cell_Info.pkl",compression="zip")


