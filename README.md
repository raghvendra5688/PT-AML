# Immunoinformatics

## Description
This repository contains all the data, code, model and results for the PT-AML project

We get input from the BeatAML project where Wave 1+2 form the training set and Wave 3+4 form the test set.
The input includes clinical, DNAseq, RNAseq, cell state, module information for each patient.
For each patient, we have information about several drugs used for treatment including the IC50 score.


## Objectives
1. Build a model to learn the intricate relationship between omics, disease state and drug response.
2. Predict the right drug for a given patient with multi-omics profile + disease state information.
3. Given a drug (new) find the right cohort of patients for which the drug will induce best response.


## Goals
1. Extract clinical, transcriptomic, dnaseq, celltype, module features common to Wave 1+2 (train) and Wave 3+4 (test).
2. Compare the multi-omics features between train and test through 2d-embedding using t-SNE to show uniformity of representations.
3. Learn feature representations for drugs using molecular fingerprint or a SMILES encoder (from drugrepurpose paper i.e. Mall et al).
4. Estimate hallmark pathway enrichment relevant for cancer using GSVA technique.
5. Estimate sample specific distance of pathways from a drug using String PPI, random walk approach and sample transcriptomic data together in a ranking setting (AUC).
6. Perform comprehensive comparison of machine learning models on features of drug and samples combined.
7. Perform explanability analysis using SHAPley method.

## Deadlines
- Project Proposal: [Jun 2023]
- Code Check in: [August 2023]
- Manuscript Draft: [September 2024]

## Scripts
1. analyze_cell_lines_rnaseq.R: Loads gene expression and clinical information from Wave 1+2 and Wave 3+4, perform t-SNE on gene expression data, merge gene expression, clinical annotations and t-SNE co-ordinates. Add pathway enrichments, cell type enrichments and module enrichments for each sample. Finally, oncogenes and 150 most variable genes are identified and only expression of those genes are considered for gene expression profile. The training and test data are saved as Traiing_Set_with_Onco_Var_Expr_Clin_PA_CTS.csv and Test_Set_with_Onco_Var_Expr_Clin_PA_CTS.csv. The metadata is saved in TRAINING_METADATA.csv file in the Results folder. Figures generated using this script include Supp. Fig 1A.

2. gene_expression_data_analysis.R: Performs variance analysis of gene expression data and identifies the optimal set of genes to select which divide the data into optimal number of clusters. Figures generated using this script include Supp. Fig 1B,C and D.

3. analyze_cell_lines_dnaseq.R: Gets the dnaseq data and divides into training and test set and converts the dnaseq data into sample by mutation type matrix for training and test sets.

4. preprocess_drugs.py: Get list of drug names from BeatAML dataset and corresponidng compound id from pubchempy package in python and get properties including InChiKey, SMILES, XLogP, Molecular Weight. Run the SMILES based encoding representation by calling 'ls_generator2.py' and save it in Drug_Full_SMILES_Embedding.csv. Figure 3A is representation of the embedding vector generation process.

5. morgan_fps.py: Get molecular fingerprints for all drugs from BeatAML dataset

6. analyze_cell_lines_drug_combinations.R: Takes input sample info, drug info, train + test feature files. Based on drug targets and string ppi network estimates propagation/diffusion score for each drug. Combines the diffusion score with the gene expression profile in the sample and then estimates distance of each pathway (geneset) from the drug using AUCell package. Final output is Revised_Train_Set_with_IC50.csv and Revised_Test_Set_with_IC50.csv. Figure 3B is representative of the random walk with restart based diffusion score combined with expression profile of sample. This is followed by AUCell for each pathway to estimate the distance of the pathway from a given drug.

7. preprocess_cellline.py: Cleans the sample features including clinical, gene expression, mutation matrix and generates training and test pickle sets.

8. make_fig1b.R: Make a complexheatmap of the oncogenes + top varying genes + clinical characteristics (from BeatAML) + pathway enrichments + celltype enrichments + module enrichments + mutation type matrix. Figure 1 is generated through this script.

9. 


 
