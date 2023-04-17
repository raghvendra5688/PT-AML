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

#Load the samples by gene, clinical, pathway enrichment, cell type enrichment data frame
train_part1_df = pd.read_csv("../Data/Training_Set_with_Expr_Clin_PA_CTS_P1.csv.gz",sep="\t")
train_part2_df = pd.read_csv("../Data/Training_Set_with_Expr_Clin_PA_CTS_P2.csv.gz",sep="\t")
test_df = pd.read_csv("../Data/Test_Set_with_Expr_Clin_PA_CTS.csv.gz",sep="\t")
train_part1_df.head()
print(train_part1_df.shape)
print(train_part2_df.shape)

# +
#Get the column names and useful columns
all_columns = list(train_part1_df.columns)

#Sample ids
sample_names = all_columns[0]
print(sample_names)

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
print(pathway_names)

#Get the information about celltypes and modules
cts_names = all_columns[22995:23015]
print(cts_names)

#Print all columns of interest
all_cols_of_interest = [sample_names]+gene_names+clin_trait_of_use+pathway_names+cts_names
print(all_cols_of_interest)

# -

len(all_cols_of_interest)

# +
#Make the big combined training and test dataframe
big_train_df = pd.concat([train_part1_df,train_part2_df],axis=0)
big_train_df = pd.DataFrame(big_train_df[all_cols_of_interest])
big_train_df.shape
big_test_df = pd.DataFrame(test_df[all_cols_of_interest])
big_test_df.shape

#Write the data frames as pickle files
big_train_df.to_pickle("../Data/Training_Set_Mod.pkl", compression="zip")
big_test_df.to_pickle("../Data/Test_Set_Mod.pkl",compression="zip")
# -


