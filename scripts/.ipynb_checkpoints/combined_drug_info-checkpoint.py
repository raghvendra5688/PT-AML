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

import numpy as np
import pandas as pd
import matplotlib
import rdkit
import pubchempy as pcp
import os

input_drug_df = pd.read_csv("../Results/Drug_Full_Modified_Info.csv")
print(input_drug_df)

#Convert the drug smiles into format ingestible for the ls_generator
src = input_drug_df["CanonicalSMILES"].tolist()
smiles_df = [src,src]
smiles_df = pd.DataFrame(smiles_df)
smiles_df = pd.DataFrame.transpose(smiles_df)
smiles_df.columns = ['src','trg']
smiles_df.to_csv("../Data/SMILES_Autoencoder/test_smiles.csv", index=False)

command = "python ls_generator2.py --input test_smiles.csv --output test_LS.csv"
os.system(command)

#Read the SMILES presentation generated and write the same
smiles_embedding_df = pd.read_csv("../Data/SMILES_Autoencoder/test_LS.csv")
final_drug_df = pd.concat([input_drug_df,smiles_embedding_df],axis=1)
final_drug_df.to_csv("../Results/Drug_Full_SMILES_Embedding.csv",index=False)


