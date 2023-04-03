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
N = nrow(train_cell_lines_mat_with_expr_clin_pa_cts)
mid_point = N/2
write.table(train_cell_lines_mat_with_expr_clin_pa_cts[c(1:mid_point),], file="Data/Training_Set_with_Expr_Clin_PA_CTS_P1.csv",row.names=F, col.names=T, quote=F, sep="\t")
write.table(train_cell_lines_mat_with_expr_clin_pa_cts[c((mid_point+1):N),], file="Data/Training_Set_with_Expr_Clin_PA_CTS_P2.csv",row.names=F, col.names=T, quote=F, sep="\t")
write.table(test_cell_lines_mat_with_expr_clin_pa_cts, file="Data/Test_Set_with_Expr_Clin_PA_CTS.csv",row.names=F, col.names=T, quote=F, sep="\t")


# #Get the mutation information for each cell line, gene combination
# ################################################################################
# mutational_info <- fread("Data/gene_cellline_somatic_mutation_edges.txt",header=T)
# mutational_info <- as.data.frame(mutational_info)
# mutational_info$CellLine <- gsub(pattern="-",replacement = "",mutational_info$CellLine)
# mutational_info$CellLine <- gsub(pattern=" ",replacement = "",mutational_info$CellLine)
# mutational_info$CellLine <- gsub(pattern="_",replacement = "",mutational_info$CellLine)
# mutational_info$CellLine <- gsub(pattern="\\/",replacement = "",mutational_info$CellLine)
# unique_cell_lines <- unique(mutational_info$CellLine)
# genes_of_interest_present <- genes_of_interest[genes_of_interest %in% colnames(inflammasome_expr_scaled_cell_lines_COSMIC_mat)]
# subset_mutational_info <- mutational_info[mutational_info$CellLine %in% rev_cell_lines_moi_COSMIC$stripped_cell_line_name & mutational_info$GeneSym %in% genes_of_interest_present,]
# 
# #Make the mutation matrix
# mutation_matrix <- Matrix(0, nrow=nrow(rev_cell_lines_moi_COSMIC), ncol=ncol(inflammasome_expr_scaled_cell_lines_COSMIC_mat))
# colnames(mutation_matrix) <- colnames(inflammasome_expr_scaled_cell_lines_COSMIC_mat)
# for (i in 1:nrow(rev_cell_lines_moi_COSMIC))
# {
#   temp <- rep(0, ncol(inflammasome_expr_scaled_cell_lines_COSMIC_mat))
#   sub_mutational_info <- subset_mutational_info[subset_mutational_info$CellLine %in% rev_cell_lines_moi_COSMIC$stripped_cell_line_name[i],]
#   if (nrow(sub_mutational_info)>0)
#   {
#     for (j in 1:nrow(sub_mutational_info))
#     {
#       gene_name <- sub_mutational_info$GeneSym[j]
#       weight <- sub_mutational_info$weight[j]
#       mutation_matrix[i,gene_name] <- weight
#     }
#   }
#   else{
#     mutation_matrix[i,] <- temp 
#   }
# }
# rownames(mutation_matrix) <- rev_cell_lines_moi_COSMIC$stripped_cell_line_name
# mutation_matrix <- as.matrix(mutation_matrix)
# colnames(mutation_matrix) <- paste0("Mutation_",colnames(mutation_matrix))
# 
# #Make the CNV matrix
# ################################################################################
# cnv_info <- fread("Data/gene_cellline_cnv_edges.txt",header=T)
# cnv_info <- as.data.frame(cnv_info)
# cnv_info[cnv_info$CellLine=="TT",]$CellLine <- "TDOTT"
# cnv_info$CellLine <- gsub(pattern="-",replacement = "",cnv_info$CellLine)
# cnv_info$CellLine <- gsub(pattern=" ",replacement = "",cnv_info$CellLine)
# cnv_info$CellLine <- gsub(pattern="_",replacement = "",cnv_info$CellLine)
# cnv_info$CellLine <- gsub(pattern="\\/",replacement = "",cnv_info$CellLine)
# unique_cell_lines <- unique(cnv_info$CellLine)
# subset_cnv_info <- cnv_info[cnv_info$CellLine %in% rev_cell_lines_moi_COSMIC$stripped_cell_line_name & cnv_info$GeneSym %in% genes_of_interest_present,]
# 
# #Make the cnv matrix
# cnv_matrix <- Matrix(0, nrow=nrow(rev_cell_lines_moi_COSMIC), ncol=ncol(inflammasome_expr_scaled_cell_lines_COSMIC_mat))
# colnames(cnv_matrix) <- colnames(inflammasome_expr_scaled_cell_lines_COSMIC_mat)
# for (i in 1:nrow(rev_cell_lines_moi_COSMIC))
# {
#   temp <- rep(0, ncol(inflammasome_expr_scaled_cell_lines_COSMIC_mat))
#   sub_cnv_info <- subset_cnv_info[subset_cnv_info$CellLine %in% rev_cell_lines_moi_COSMIC$stripped_cell_line_name[i],]
#   if (nrow(sub_cnv_info)>0)
#   {
#     for (j in 1:nrow(sub_cnv_info))
#     {
#       gene_name <- sub_cnv_info$GeneSym[j]
#       weight <- sub_cnv_info$weight[j]
#       cnv_matrix[i,gene_name] <- weight
#     }
#   }
#   else{
#     cnv_matrix[i,] <- temp 
#   }
# }
# rownames(cnv_matrix) <- rev_cell_lines_moi_COSMIC$stripped_cell_line_name
# cnv_matrix <- as.matrix(cnv_matrix)
# colnames(cnv_matrix) <- paste0("CNV_",colnames(cnv_matrix))
# 
# #Combine the cell line meta data + pathway activity + immune cell activity + all gene expression
# ###############################################################################
# combined_COSMIC_cell_line_df <- as.data.frame(cbind(rev_cell_lines_moi_COSMIC,
#                                                     rev_ssgsea_results, 
#                                                     mutation_matrix, 
#                                                     cnv_matrix,
#                                                     inflammasome_expr_scaled_cell_lines_COSMIC_mat))
# 
# #Make tSNE plot with the modified dataset for each cell line that we have
# mod_tsne_out <- Rtsne(X=as.matrix(combined_COSMIC_cell_line_df[,c(11:ncol(combined_COSMIC_cell_line_df))]), dims=2, perplexity = 5.0, pca_center = F, pca_scale = F, pca = F)
# mod_tsne_df <- as.data.frame(mod_tsne_out[["Y"]])
# mod_tsne_df$Pheno <- rev_cell_lines_moi_COSMIC$lineage
# mod_unique_phenotypes <- unique(mod_tsne_df$Pheno)
# mod_tsne_df$Pheno <- factor(mod_tsne_df$Pheno, levels = mod_unique_phenotypes)
# 
# g_tsne_mod <- ggplot(data=mod_tsne_df, aes(x=V1, y=V2, color=Pheno)) + geom_point(aes(color=Pheno), size=2) + theme_bw() + xlab("T-SNE Dim1") + ylab("T-SNE Dim2") +
#   scale_color_manual(name="Sample",labels = mod_unique_phenotypes, values = as.vector(P28))+
#   theme(legend.title.align=0.0, legend.text.align = 0.0) +
#   theme(axis.text.x = element_text(color = "grey20", size = 16, angle = 0, hjust = .5, vjust = .5, face = "plain"),
#         strip.text = element_text(color = "white", size=20, angle=0, hjust = 0.5, vjust = 0.5, face = "plain"),
#         axis.text.y = element_text(color = "grey20", size = 16, angle = 0, hjust = 1, vjust = 0, face = "plain"),
#         axis.title.x = element_text(color = "grey20", size = 20, angle = 0, hjust = .5, vjust = 0, face = "plain"),
#         axis.title.y = element_text(color = "grey20", size = 20, angle = 90, hjust = .5, vjust = .5, face = "plain"),
#         legend.text = element_text(color="grey20", size=18, angle = 0, face="plain"),
#         title=element_text(color="grey20", size=20, face="plain"))
# ggsave(filename=paste0("results/Mod_T-SNE_plot.pdf"), plot = g_tsne_mod, device = pdf(), height = 6, width=12, units="in")
# dev.off()
# 
# #Remove columns with no variance
# col_ids_to_remove <- which(unlist(lapply(combined_COSMIC_cell_line_df[,c(11:ncol(combined_COSMIC_cell_line_df))],sd))==0)+10
# rev_combined_COSMIC_cell_line_df <- combined_COSMIC_cell_line_df[,-c(col_ids_to_remove)]
# 
# #Make the metadata information to save and use for ablation studies
# metadata_info <- as.data.frame(cbind(c(1:ncol(combined_COSMIC_cell_line_df)),colnames(combined_COSMIC_cell_line_df)))
# colnames(metadata_info) <- c('Index','Columns')
# metadata_info$Index <- as.numeric(as.vector(metadata_info$Index))
# metadata_info$Columns <- as.character(as.vector(metadata_info$Columns))
# metadata_info$Type <- "Clinical_Features"
# metadata_info[11:17,]$Type <- "Pathway_Features"
# metadata_info[c(18:185),]$Type <- "Mutation_Features"
# metadata_info[c(186:353),]$Type <- "CNV_Features"
# metadata_info[c(354:521),]$Type <- "Expr_Features"
# rev_metadata_info  <- metadata_info[-c(col_ids_to_remove),]
# 
# write.table(rev_metadata_info, file = "Data/Cell_Lines_Metdata_Description_Inflammasome.csv", row.names=T, col.names=T, quote=F, sep="\t")
# write.table(rev_combined_COSMIC_cell_line_df, file="Data/Cell_Lines_Metadata_Pathway_Activities_PANoptosis_Markers_Expr_Inflammasome.csv", col.names=T, row.names=F, sep="\t", quote=F)
# write.table(scaled_cell_lines_COSMIC_mat, file="Data/Cell_Lines_Full_Scaled_Expr.csv",row.names=T, col.names=T, sep="\t", quote=F)
# 
# #Convert pathway geneset to an edgelist
# pathway_gene_edgelist <- NULL
# for (i in 1:length(Selected.pathways))
# {
#   pathway_name <- names(Selected.pathways)[i]
#   geneset <- Selected.pathways[[i]]
#   temp <- cbind(rep(pathway_name,length(geneset)),geneset)
#   pathway_gene_edgelist <- rbind(pathway_gene_edgelist, temp)
# }
# pathway_gene_edgelist <- as.data.frame(pathway_gene_edgelist)
# colnames(pathway_gene_edgelist) <- c("Pathway","Gene")
# write.table(pathway_gene_edgelist, file="Data/Pathways_Genes_Edgelist_Inflammasome.csv", row.names=F, col.names=T, quote=F, sep="\t")