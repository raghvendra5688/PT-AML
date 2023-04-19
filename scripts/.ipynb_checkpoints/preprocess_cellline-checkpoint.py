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

import numpy as np
import pandas as pd
import pyreadr

#Load the samples by gene, clinical, pathway enrichment, cell type enrichment data frame
train_part1_df = pd.read_csv("../Data/Revised_Training_Set_with_Expr_Clin_PA_CTS_P1.csv.gz",sep="\t",low_memory=False)
train_part2_df = pd.read_csv("../Data/Revised_Training_Set_with_Expr_Clin_PA_CTS_P2.csv.gz",sep="\t",low_memory=False)
test_df = pd.read_csv("../Data/Revised_Test_Set_with_Expr_Clin_PA_CTS.csv.gz",sep="\t",low_memory=False)
train_part1_df.head()
print(train_part1_df.shape)
print(train_part2_df.shape)

#Load the mutation information
out = pyreadr.read_r("../Data/Train_Test_Mutation_Matrices.Rdata")
train_mut_df = out["train_mut_mat"]
test_mut_df = out["test_mut_mat"]
train_mut_var_df = out["train_mut_var_mat"]
test_mut_var_df = out["test_mut_var_mat"]
print(train_mut_df.shape)
print(test_mut_df.shape)
train_mut_df.columns

# +
#Get the column names and useful columns
all_columns = list(train_part1_df.columns)

#Sample ids
sample_names = all_columns[0]

#Gene names
gene_names = all_columns[1:22844]

#Clinical traits with T-sne
clin_traits = all_columns[22844:22941]
clin_trait_of_use = ['Tsne1','Tsne2','consensus_sex','ageAtDiagnosis','diseaseStageAtSpecimenCollection','vitalStatus',
                     'overallSurvival', '%.Blasts.in.BM', '%.Blasts.in.PB', '%.Eosinophils.in.PB', '%.Lymphocytes.in.PB', 
                     '%.Monocytes.in.PB', '%.Neutrophils.in.PB','ALT', 'AST', 'albumin', 'creatinine', 
                     'hematocrit', 'hemoglobin','plateletCount','wbcCount']

#A description of the min max values
train_part1_df[clin_trait_of_use].describe()

#Get the information about pathways
pathway_names = all_columns[22941:22995]

#Get the information about celltypes and modules
cts_names = all_columns[22995:23015]

#Print all columns of interest
all_cols_of_interest = [sample_names]+gene_names+clin_trait_of_use+pathway_names+cts_names
print(all_cols_of_interest)
# -

len(all_cols_of_interest)

# +
#Make the big combined training and test dataframe
big_train_df = pd.concat([train_part1_df,train_part2_df],axis=0)
big_train_df = pd.DataFrame(big_train_df[all_cols_of_interest])
big_test_df = pd.DataFrame(test_df[all_cols_of_interest])

#Join the training dataframe with mutation information
big_train_df = pd.merge(big_train_df,train_mut_df,on='dbgap_rnaseq_sample')
big_train_df = pd.merge(big_train_df, train_mut_var_df, on="dbgap_rnaseq_sample")

big_test_df = pd.merge(big_test_df, test_mut_df, on="dbgap_rnaseq_sample")
big_test_df = pd.merge(big_test_df, test_mut_var_df, on="dbgap_rnaseq_sample")
print(big_train_df.shape)
print(big_test_df.shape)
sum(big_train_df.columns==big_test_df.columns)
# -
#Write the data frames as pickle files
big_train_df.to_pickle("../Data/Training_Set_Mod.pkl", compression="zip")
big_test_df.to_pickle("../Data/Test_Set_Mod.pkl",compression="zip")



