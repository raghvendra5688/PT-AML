library(data.table)
library(ggplot2)
library(gplots)
library(ComplexHeatmap)
library(BiocManager)
library(GSVA)
library(BiocParallel)
library(doParallel)
library(stringr)
library(circlize)
library(RColorBrewer)
library(ggpubr)
library(stringr)
library(GSA)
library(Matrix)
library(extrafont)
library(Rtsne)
library(immunedeconv)
library(Polychrome)
library(R.utils)
loadfonts()
registerDoParallel(20)
ht_opt$message = FALSE

setwd("/home/raghvendra/TII/Projects/Raghav/Immunoinformatics/")
source("scripts/all_cells_functions.R")

################################################################################
#Get the RNAseq data from all the cell lines and clean up to assemble one full RNAseq matrix
cell_lines_aa_df <- fread("BeatAML/Data/beataml_waves1to4_norm_exp_dbgap_partaa.gz",header=T)
cell_lines_ab_df <- fread("BeatAML/Data/beataml_waves1to4_norm_exp_dbgap_partab.gz",header=F)
cell_lines_ac_df <- fread("BeatAML/Data/beataml_waves1to4_norm_exp_dbgap_partac.gz",header=F)
cell_lines_aa_df <- as.data.frame(cell_lines_aa_df)
cell_lines_ab_df <- as.data.frame(cell_lines_ab_df)
cell_lines_ac_df <- as.data.frame(cell_lines_ac_df)
colnames(cell_lines_ab_df) <- colnames(cell_lines_aa_df)
colnames(cell_lines_ac_df) <- colnames(cell_lines_aa_df)
cell_lines_df <- rbind(cell_lines_aa_df,cell_lines_ab_df,cell_lines_ac_df)
rm(cell_lines_aa_df, cell_lines_ab_df, cell_lines_ac_df)
gc()

gene_names <- cell_lines_df$display_label
sample_names <- colnames(cell_lines_df)[c(5:ncol(cell_lines_df))]

#Convert the dataframe into a matrix with gene expression as rows and sample names as columns
cell_lines_mat <- as.matrix(cell_lines_df[,c(5:ncol(cell_lines_df))])
rownames(cell_lines_mat) <- gene_names
colnames(cell_lines_mat) <- sample_names
rm(cell_lines_df)
gc()

################################################################################
#Get the cell line meta annotation to get information about all samples with RNAseq and in which wave they belonged
cell_lines_metadata <- fread("BeatAML/Data/beataml_wv1to4_clinical.csv",header=T)
cell_lines_metadata <- as.data.frame(cell_lines_metadata)
training_cohort <- c("Waves1+2","Both")
test_cohort <- c("Waves3+4")
#We consider samples with both rna and dna data
training_cohort_samples <- setdiff(c(cell_lines_metadata[cell_lines_metadata$cohort %in% training_cohort,]$dbgap_rnaseq_sample,
                                     cell_lines_metadata[cell_lines_metadata$cohort %in% training_cohort,]$dbgap_dnaseq_sample),"")
test_cohort_samples <- setdiff(c(cell_lines_metadata[cell_lines_metadata$cohort %in% test_cohort,]$dbgap_rnaseq_sample,
                                 cell_lines_metadata[cell_lines_metadata$cohort %in% test_cohort,]$dbgap_dnaseq_sample),
                                 "")
#These samples should all have rna-seq and hence should have their names in the cell_lines_mat 
rev_training_cohort_samples <- training_cohort_samples[training_cohort_samples %in% sample_names]
rev_test_cohort_samples <- test_cohort_samples[test_cohort_samples %in% sample_names]

################################################################################
#Make the t-SNE plot using full cell line dataset
rev_cell_lines_mat <- cell_lines_mat[,c(rev_training_cohort_samples, rev_test_cohort_samples)]
rev_cell_lines_metadata <- cell_lines_metadata[cell_lines_metadata$dbgap_rnaseq_sample %in% c(rev_training_cohort_samples,rev_test_cohort_samples),]
rm(cell_lines_mat)
gc()

set.seed(123)
full_expr_tsne_out <- Rtsne(X=t(rev_cell_lines_mat), dims=2, perplexity = 20.0, pca_center = F, pca_scale = F, pca = F)
full_expr_tsne_df <- as.data.frame(full_expr_tsne_out[["Y"]])
colnames(full_expr_tsne_df) <- c("Tsne1","Tsne2")
full_expr_tsne_df$Pheno <- c(rep("Train",length(rev_training_cohort_samples)),rep("Test",length(rev_test_cohort_samples)))
unique_phenotypes <- unique(full_expr_tsne_df$Pheno)
full_expr_tsne_df$Pheno <- factor(full_expr_tsne_df$Pheno, levels = unique_phenotypes)
full_expr_tsne_df$dbgap_rnaseq_sample <- colnames(rev_cell_lines_mat)
full_expr_tsne_with_annotations_df <- merge(full_expr_tsne_df, rev_cell_lines_metadata, by="dbgap_rnaseq_sample",all = TRUE)
write.table(full_expr_tsne_with_annotations_df, file = "Data/Train_Test_Tsne_with_Annotations.csv",col.names = T, row.names=F, quote=F, sep=";")

# create your own color palette based on `seedcolors`
P28 = createPalette(28,  c("#ff0000", "#00ff00", "#0000ff"))
palette(P28)
g_tsne <- ggplot(data=full_expr_tsne_with_annotations_df, aes(x=Tsne1, y=Tsne2, color=Pheno)) + geom_point(aes(color=Pheno), size=2) + theme_bw() + xlab("T-SNE Dim1") + ylab("T-SNE Dim2") +
  #scale_color_manual(name="Sample",labels = unique_phenotypes, values = as.vector(P28) )+
  theme(legend.title.align=0.0, legend.text.align = 0.0) +
  theme(axis.text.x = element_text(color = "grey20", size = 16, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        strip.text = element_text(color = "white", size=20, angle=0, hjust = 0.5, vjust = 0.5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 16, angle = 0, hjust = 1, vjust = 0, face = "plain"),
        axis.title.x = element_text(color = "grey20", size = 20, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 20, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        legend.text = element_text(color="grey20", size=18, angle = 0, face="plain"),
        title=element_text(color="grey20", size=20, face="plain"))
ggsave(filename=paste0("Results/Full_Expr_T-SNE_plot.pdf"), plot = g_tsne, device = pdf(), height = 8, width=8, units="in")
dev.off()

################################################################################
#Build the cell line dataset with clinical annotations and gene expression profiles
rev_cell_lines_mat <- as.data.frame(t(rev_cell_lines_mat))
rev_cell_lines_mat$dbgap_rnaseq_sample <- rownames(rev_cell_lines_mat)

final_cell_lines_mat <- merge(rev_cell_lines_mat, full_expr_tsne_with_annotations_df, by="dbgap_rnaseq_sample",all=TRUE)

################################################################################
#Get cancer associated pathways and panoptosis pathway with its activity for each cell lines
load("Data/Selected.pathways.3.4.RData")

rev_hallmark_pathways_of_interest <- Selected.pathways
rev_ssgsea_results <- as.data.frame(t(gsva(expr = as.matrix(t(rev_cell_lines_mat[,-ncol(rev_cell_lines_mat)])), gset.idx.list = rev_hallmark_pathways_of_interest, method="ssgsea", kcdf = "Gaussian", parallel.sz=20)))
rev_ssgsea_results$dbgap_rnaseq_sample <- rownames(rev_ssgsea_results)

#Merge the pathway activity matrix with the cell lines matrix
cell_lines_mat_with_expr_clinical_pathway_activity <- merge(final_cell_lines_mat, rev_ssgsea_results, by="dbgap_rnaseq_sample", all=TRUE)

################################################################################
#Get the marker gene set for cell type and estimate enrichment for each cell type
celltype_sig <- parse.van.galen.supp("BeatAML/Data/1-s2.0-S0092867419300947-mmc3.xlsx")
celltypes_of_interest <- c("cDC-like","GMP-like","HSC-like","Monocyte-like","Progenitor-like","Promono-like")
rev_celltype_sig <- celltype_sig[celltype_sig$vg_type %in% celltypes_of_interest,]
celltype_genelist <- NULL
unique_celltypes <- unique(rev_celltype_sig$vg_type)
for (i in 1:length(unique_celltypes))
{
  celltype_name <- unique_celltypes[i]
  geneset <- rev_celltype_sig[rev_celltype_sig$vg_type==celltype_name,]$display_label
  celltype_genelist[[i]] <- geneset
}
names(celltype_genelist) <- unique_celltypes

#Make the gene set enrichment analysis to get scores for each celltype
ssgsea_celltype_results <- as.data.frame(t(gsva(expr=as.matrix(t(rev_cell_lines_mat[,-ncol(rev_cell_lines_mat)])), gset.idx.list = celltype_genelist, method="ssgsea", kcdf = "Gaussian", parallel.sz=20)))

#Get the marker gene set for the modules and enrichment scores
load("BeatAML/Data/merged_older_wgcna_kme.RData")
unique_modules <- unique(cur.map$cur_labels)
module_genelist <- NULL
for (i in 1:length(unique_modules))
{
  module_name <- unique_modules[i]
  genelist <- cur.map[cur.map$cur_labels==module_name,]$display_label
  module_genelist[[i]] <- genelist
}
names(module_genelist) <- unique_modules

ssgsea_module_results <- as.data.frame(t(gsva(expr=as.matrix(t(rev_cell_lines_mat[,-ncol(rev_cell_lines_mat)])), gset.idx.list = module_genelist, method="ssgsea", kcdf = "Gaussian", parallel.sz=20)))

#Combine the ssgsea results for celltypes and modules
ssgsea_celltype_module_combinations <- as.data.frame(cbind(ssgsea_celltype_results,ssgsea_module_results))
ssgsea_celltype_module_combinations$dbgap_rnaseq_sample <- rownames(ssgsea_celltype_module_combinations)

#Merge the cell type enrichment matrix with cell lines matrix
cell_lines_mat_with_expr_clin_pa_cts <- merge(cell_lines_mat_with_expr_clinical_pathway_activity, ssgsea_celltype_module_combinations,
                                              by = "dbgap_rnaseq_sample", all=TRUE)

################################################################################
#Divide data into train and test set and write it down
train_cell_lines_mat_with_expr_clin_pa_cts <- cell_lines_mat_with_expr_clin_pa_cts[cell_lines_mat_with_expr_clin_pa_cts$dbgap_rnaseq_sample %in% training_cohort_samples,]
test_cell_lines_mat_with_expr_clin_pa_cts <- cell_lines_mat_with_expr_clin_pa_cts[cell_lines_mat_with_expr_clin_pa_cts$dbgap_rnaseq_sample %in% test_cohort_samples,]
train_sample_ids <- train_cell_lines_mat_with_expr_clin_pa_cts$dbgap_rnaseq_sample
test_sample_ids <- test_cell_lines_mat_with_expr_clin_pa_cts$dbgap_rnaseq_sample
save(train_sample_ids,test_sample_ids,file='Data/Train_Test_Ids.Rdata')
N = nrow(train_cell_lines_mat_with_expr_clin_pa_cts)
mid_point = N/2
write.table(train_cell_lines_mat_with_expr_clin_pa_cts[c(1:mid_point),], file="Data/Training_Set_with_Expr_Clin_PA_CTS_P1.csv",row.names=F, col.names=T, quote=F, sep="\t")
write.table(train_cell_lines_mat_with_expr_clin_pa_cts[c((mid_point+1):N),], file="Data/Training_Set_with_Expr_Clin_PA_CTS_P2.csv",row.names=F, col.names=T, quote=F, sep="\t")
write.table(test_cell_lines_mat_with_expr_clin_pa_cts, file="Data/Test_Set_with_Expr_Clin_PA_CTS.csv",row.names=F, col.names=T, quote=F, sep="\t")

#Write the dataset with only oncogenes
################################################################################
list_oncogenes <- fread("Data/Oncogenes.csv",header=T)
oncogenes_symbol <- list_oncogenes$`Gene Symbol`
oncogene_ids <- which(colnames(train_cell_lines_mat_with_expr_clin_pa_cts) %in% oncogenes_symbol)
train_cell_lines_mat_with_onco_expr_clin_pa_cts <- train_cell_lines_mat_with_expr_clin_pa_cts[,c(1,oncogene_ids,c(22844:ncol(train_cell_lines_mat_with_expr_clin_pa_cts)))]
test_cell_lines_mat_with_onco_expr_clin_pa_cts <- test_cell_lines_mat_with_expr_clin_pa_cts[,c(1,oncogene_ids,c(22844:ncol(test_cell_lines_mat_with_expr_clin_pa_cts)))]
write.table(train_cell_lines_mat_with_onco_expr_clin_pa_cts,file="Data/Training_Set_with_Onco_Expr_Clin_PA_CTS.csv",row.names=F, col.names=T, quote=F, sep="\t")
write.table(test_cell_lines_mat_with_onco_expr_clin_pa_cts,file="Data/Test_Set_with_Onco_Expr_Clin_PA_CTS.csv",row.names=F, col.names=T, quote=F, sep="\t")


#Calculate variance on training set
##############################################################################
train_part1_df <- fread("Data/Revised_Training_Set_with_Expr_Clin_PA_CTS_P1.csv.gz",header=T,sep="\t")
train_part2_df <- fread("Data/Revised_Training_Set_with_Expr_Clin_PA_CTS_P2.csv.gz",header=T,sep="\t")
train_part1_df <- as.data.frame(train_part1_df)
train_part2_df <- as.data.frame(train_part2_df)
train_df <- as.data.frame(rbind(train_part1_df,train_part2_df))
var_list <- lapply(train_df[,c(2:22844)],var)
variance_genes <- as.numeric(var_list)
names(variance_genes) <- names(var_list)
variance_genes <- sort(variance_genes,decreasing=T)

column_names <- colnames(train_df)
write.table(column_names,file="Results/TRAINING_Metadata.csv",row.names=T,col.names=F,quote=F,sep="\t")

train_df[,c(23001:23015)]