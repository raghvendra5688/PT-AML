#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import pandas as pd
import matplotlib
import rdkit
import pubchempy as pcp


# %%
#Read the data file which contains the name of all drugs
combined_df = pd.read_csv("../BeatAML/Data/beataml_probit_curve_fits_v4_dbgap.txt", delimiter="\t")


# %%
#Get the names of all the drugs and put it in a csv file
drug_names_list = combined_df["inhibitor"].unique().tolist()
fp = open("../Results/Drug_Names.csv","w")
fp.write("Drug_Names"+"\n")
for drug in drug_names_list:
    outstr = drug+"\n"
    fp.write(outstr)
fp.close()


# %%
#Get the drug - and its corresponidng cids into a dictionary
drug_cid_dict = {}
for drug in drug_names_list:
    rev_drug_name = drug.split(" ")[0]
    cids = pcp.get_cids(rev_drug_name,'name','substance',list_return='flat')
    if (len(cids)>0):
        #Put the original name of the drug with the first cid encountered in Pubchem
        drug_cid_dict[drug] = cids[0]
    else:
        drug_cid_dict[drug] = []

# %%
final_drug_info = []
for drug in drug_names_list:
    if (drug_cid_dict[drug]!=[]):
        tmp_compound_dict = pcp.get_properties(['InChIKey','CanonicalSMILES','XLogP','MolecularWeight'],
                                           drug_cid_dict[drug],'cid')
        tmp_compound_dict[0]['Name'] = drug
        final_drug_info.append(tmp_compound_dict[0])
    else:
        final_drug_info.append({'CID':None, 'CanonicalSMILES':None, 'InChIKey': None, 'XLogP':None,
                               'MolecularWeight': None, 'Name': drug})

# %%
#Convert a list of dictionaries into a dataframe
final_drug_info_df = pd.DataFrame(final_drug_info)
print(final_drug_info_df)
final_drug_info_df.to_csv("../Results/Drug_Full_Info.csv",sep=",",index=False)

# %%
