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
import matplotlib
import rdkit

#Read the data file which contains the name of all drugs
combined_df = pd.read_csv("../BeatAML/Data/beataml_probit_curve_fits_v4_dbgap.txt", delimiter="\t")
print(combined_df)


