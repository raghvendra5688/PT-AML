library(data.table)
library(ggplot2)
library(stringi)
library(stringr)
library(igraph)
library(matlib)
library(mnormt)
library(Matrix)
library(diffusr)
library(dnet)
library(doParallel)
library(foreach)
library(tmod)
library(AUCell)
registerDoParallel(cores=12)

setwd("/research/groups/kannegrp/home/rmall/projects/Raghav/Drug_Sensitivity_PANoptosis/")

#Get cell line information
#cell line info: sex, age, lineage, primary or metastasis, pathways, immune activity, immune concentration, gene names, mutation, copy number abberation
cell_line_info <- fread("Data/Cell_Lines_Metadata_Pathway_Activities_PANoptosis_Markers_Expr_Inflammasome.csv",header=T)
cell_line_info <- as.data.frame(cell_line_info)
revised_cell_line_info <- cell_line_info[,c(1,3,7:ncol(cell_line_info))]

#Get the drug information for unique drugs with at least 1 target and has canonical smiles
drug_info <- fread("Data/full_drug_info_gdsc_with_manual_annotation.csv",header=T)
drug_info <- as.data.frame(drug_info)
drug_info <- drug_info[complete.cases(drug_info),]
drug_info <- drug_info[drug_info$All_Targets!="" & !is.na(drug_info$Drug_CanonicalSmiles),]
drug_info <- drug_info[,c(2:ncol(drug_info))]
drug_info <- unique(drug_info)

#Get the revised drug info
full_drug_info <- NULL
unique_drugs <- unique(drug_info$Drug_Name)
for (i in 1:length(unique_drugs))
{
  drug_name <- unique_drugs[i]
  ids <- which(drug_info$Drug_Name==drug_name)
  if (length(ids)>1)
  {
    index <- 1
    temp <- drug_info[ids[index],]
  }else{
    temp <- drug_info[ids,]
  }
  full_drug_info <- rbind(full_drug_info, temp)
}
full_drug_info <- as.data.frame(full_drug_info)
colnames(full_drug_info) <- colnames(drug_info)
full_drug_info$Drug_Name <- toupper(full_drug_info$Drug_Name)
revised_drug_info <- full_drug_info[,c(1,3,5:10)]

#Get the drug cell combination information
all_sensitivity_info <- fread("Data/sanger-dose-response.csv",header=T)
all_sensitivity_info <- as.data.frame(all_sensitivity_info)
all_sensitivity_info$DRUG_NAME <- toupper(all_sensitivity_info$DRUG_NAME)
revised_sensitivity_info <- all_sensitivity_info[,c(10,11,8)]
revised_sensitivity_info$y_ic50 <- -log10(revised_sensitivity_info$IC50_PUBLISHED)
revised_sensitivity_info <- revised_sensitivity_info[revised_sensitivity_info$ARXSPAN_ID!="",]

#Change the drug and cell line info to perform inner joins
colnames(revised_drug_info)[1] <- "DRUG_NAME"
colnames(revised_cell_line_info)[1] <- "ARXSPAN_ID"

#Perform inner joins
################################################################################
refined_drug_cell_line_info <- merge(revised_sensitivity_info, revised_drug_info, by="DRUG_NAME")
refined_drug_cell_line_info <- merge(refined_drug_cell_line_info, revised_cell_line_info, by = "ARXSPAN_ID")

#Sort to remove duplicate Drug and Cell Line ids from GDSC1, GSDC2
refined_drug_cell_line_info <- refined_drug_cell_line_info[order(refined_drug_cell_line_info$ARXSPAN_ID,refined_drug_cell_line_info$DRUG_NAME),]
rownames(refined_drug_cell_line_info) <- NULL

#Get unique row ids of drug and cell line combinations
ids_to_keep <- as.numeric(rownames(unique(refined_drug_cell_line_info[,c(1,2)])))
final_drug_cell_info <- refined_drug_cell_line_info[ids_to_keep,]

#Load the cell line expression profile
################################################################################
cell_line_expr_df <- read.table("Data/Cell_Lines_Full_Scaled_Expr.csv",header=T)

#Load the string ppi matrix as an adjacency matrix
################################################################################
string_ppi_df <- read.table("Data/String_PPI_Cutoff_0.7.csv",header=T)
g_ppi <- graph_from_edgelist(as.matrix(string_ppi_df[,c(1:2)]), directed = T)
E(g_ppi)$weight <- string_ppi_df$combined_score
N_ppi <- length(V(g_ppi))

#Get the data frame of each drug and the affinity to all proteins from their target genes
################################################################################
unique_drugs <- unique(final_drug_cell_info$DRUG_NAME)
p0_matrix <- matrix(0, nrow=N_ppi, ncol=length(unique_drugs))
rownames(p0_matrix) <- V(g_ppi)$name
for (i in 1:length(unique_drugs))
{
  drug_name <- unique_drugs[i]
  targets <- final_drug_cell_info[final_drug_cell_info$DRUG_NAME==drug_name,]$All_Targets[1]
  targets <- unlist(strsplit(targets,split=";"))
  p0_vec <- rep(0, N_ppi)
  names(p0_vec) <- V(g_ppi)$name
  p0_vec[names(p0_vec) %in% targets] <- 1
  p0_matrix[,i] <- p0_vec
}
pinf_matrix <- sqrt(dRWR(g = g_ppi, normalise = "laplacian", setSeeds = p0_matrix, restart = 0.5, multicores = 12))
rownames(pinf_matrix) <- V(g_ppi)$name
colnames(pinf_matrix) <- unique_drugs
pinf_matrix <- as.matrix(pinf_matrix)

#Build the drug affinity and cell line expression combination matrix
all_genes_in_ppi <- rownames(pinf_matrix)
all_genes_in_cell_line <- colnames(cell_line_expr_df)
common_genes <- intersect(all_genes_in_ppi, all_genes_in_cell_line)

#Get the stationary affinity matrix from the drugs for all common set of genes
revised_pinf_matrix <- t(pinf_matrix[common_genes,])

#Get the expression of common set of genes
revised_cell_line_expr_df <- cell_line_expr_df[,common_genes]

#Get drug, cell line specific affinity/distance measures
drug_cell_affinity_combinations <- t(sapply(1:length(common_genes), function(i) tcrossprod(revised_pinf_matrix[,i ], revised_cell_line_expr_df[,i ])))
drug_cell_affinity_combinations_mat <- Matrix(drug_cell_affinity_combinations, sparse=T)
rm(drug_cell_affinity_combinations)
gc()

#Get the column names for all the data as well as gene names
all_drug_cell_line_combination <- as.data.frame(expand.grid(rownames(revised_pinf_matrix),rownames(revised_cell_line_expr_df)))
all_drug_cell_line_combination <- paste0(all_drug_cell_line_combination$Var1,"_",all_drug_cell_line_combination$Var2) 
colnames(drug_cell_affinity_combinations_mat) <- all_drug_cell_line_combination 
rownames(drug_cell_affinity_combinations_mat) <- common_genes

#To get the AUC for the genesets of interest to estimate enrichment
#Get the inflammasome genesets (from REACTIOME, GO BP, KEGG)
inflammasome_geneset <- read.table("Data/Inflammasome_geneset.txt",header=T)
go_inflammasome_geneset <- read.table("Data/GO_Inflammasome_geneset.txt",header=T)
nlr_geneset <- read.table("Data/NLR_geneset.txt",header=T)
pyroptosis_geneset <- read.table("Data/Pyroptosis_geneset.txt",header=T)
necroptosis_geneset <- read.table("Data/Necroptosis_geneset.txt",header=T)

#Get cancer associated pathways and panoptosis pathway with its activity for each cell lines
load("Data/Selected.pathways.3.4.RData")
load("Data/panoptosis_genes.Rdata")
icr_geneset <- c("IFNG", "TXB21", "CD8B", "CD8A", "IL12B", "STAT1", "IRF1", "CXCL9", "CXCL10", "CCL5", 
                              "GNLY", "PRF1", "GZMA", "GZMB", "GZMH", "CD274", "PDCD1", "CTLA4", "FOXP3", "IDO1")
Selected.pathways <- list()
#Selected.pathways[["[HM] Apoptosis"]] <- apoptosis_geneset
Selected.pathways[["[MRK] PANoptosis"]] <- panoptosis_full_genes
Selected.pathways[["[DB] ICD"]] <- icr_geneset
Selected.pathways[["[KG] NLR Signaling"]] <- nlr_geneset$KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY
Selected.pathways[["[RT] Inflammasome"]] <- inflammasome_geneset$REACTOME_INFLAMMASOMES
Selected.pathways[["[GO] Inflammasome Complex"]] <- go_inflammasome_geneset$GOBP_INFLAMMASOME_COMPLEX_ASSEMBLY
Selected.pathways[["[RT] Pyroptosis"]] <- pyroptosis_geneset$REACTOME_PYROPTOSIS
Selected.pathways[["[GO] Necroptosis"]] <- necroptosis_geneset$GOBP_NECROPTOTIC_SIGNALING_PATHWAY

#Build the cell rankings using AUcell method on the drug-cell line distance information
blocks = 50
N = ncol(drug_cell_affinity_combinations_mat)
blocksize = ceiling(N/50)
auc_scores_df <- NULL
for (i in 1:50)
{
  if (i!=50)
  {
    ids <- c(((i-1)*blocksize+1):(i*blocksize))
  }else{
    ids <- c(((i-1)*blocksize+1):(N))
  }
  cell_rankings_temp <- AUCell_buildRankings(exprMat = as.matrix(drug_cell_affinity_combinations_mat[,ids]), nCores = 12, plotStats = T)
  temp_auc_scores <- AUCell_calcAUC(geneSets = Selected.pathways, rankings = cell_rankings_temp, nCores = 12, normAUC = T, aucMaxRank = ceiling(0.5*nrow(cell_rankings_temp)))
  auc_scores_df <- AUCell::cbind(auc_scores_df, temp_auc_scores@assays@data@listData$AUC)
  rm(temp_auc_scores)
  rm(cell_rankings_temp)
  gc()
}
auc_scores_df <- t(auc_scores_df)
colnames(auc_scores_df) <- paste0("AUC_",names(Selected.pathways))

#Drug - Cell Line combinations to keep
combinations_to_keep <- paste0(final_drug_cell_info$DRUG_NAME,"_",final_drug_cell_info$ARXSPAN_ID)
subset_auc_scores_df <- auc_scores_df[combinations_to_keep,]
subset_auc_scores_df <- signif(subset_auc_scores_df, digits=3)
revised_final_drug_cell_info <- as.data.frame(cbind(final_drug_cell_info,subset_auc_scores_df))
colnames(revised_final_drug_cell_info) <- c(colnames(final_drug_cell_info),colnames(subset_auc_scores_df))
revised_final_drug_cell_info[,c(17:ncol(revised_final_drug_cell_info))] <- signif(revised_final_drug_cell_info[,c(17:ncol(revised_final_drug_cell_info))], digits = 3)

#Write the final output file to use for the downstream ML models
write.table(revised_final_drug_cell_info, file="Data/dose_response_with_full_data_inflammasome.csv",row.names=F, col.names=T, sep="\t", quote=F)
write.table(colnames(revised_final_drug_cell_info), file="Data/dose_response_full_metadata_inflammasome.csv",col.names=F,quote=F)

